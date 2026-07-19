from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import secrets
import stat
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import adaptive_v4_execution_environment as execution_environment
import adaptive_v4_gpu_lock as gpu_lock
import calibrate_p2_direct_soft_lag as calibration
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import run_p2_direct_training_matrix as training_matrix

EXPERIMENT_ID = "p2-post-rank-direct-soft-lag-calibration-matrix-v1.2"
ARTIFACT_TYPE = "direct-soft-lag-calibration-matrix"
SCHEMA_VERSION = 6

FROZEN_SCALES = ("s55", "s151")
FROZEN_TRAINING_SEEDS = (6071406, 6071407, 6071408, 6071409, 6071410)
FROZEN_CALIBRATION_SEEDS = (7071406, 7071407, 7071408, 7071409, 7071410)
EXPECTED_CELLS = len(FROZEN_SCALES) * len(FROZEN_TRAINING_SEEDS)

TRAINING_OUTPUT_ROOT = training_matrix.OUTPUT_ROOT
OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/calibration-v1-2")
MATRIX_SUMMARY_NAME = "calibration-matrix-v1-2.summary.json"
MATRIX_SUMMARY = OUTPUT_ROOT / MATRIX_SUMMARY_NAME
SUPERSEDED_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/calibration"
)
SUPERSEDED_MATRIX_SUMMARY_NAME = "calibration-matrix.summary.json"
RETRY_ADMISSION_NAME = "calibration-v1-2-retry-admission.json"
RETRY_ADMISSION_STAGING_NAME = ".p2-direct-calibration-v1-2-admission.pending"
RETRY_ADMISSION_ID = "p2-direct-calibration-one-shot-retry-admission-v1.2"
RETRY_ADMISSION_REASON = "checkpoint-path-spelling-parent-validator-false-negative-v1"
RETRY_COORDINATE = ("s55", 6071406, 7071406, 10071406)
REQUIRE_RETRY_ADMISSION = True
RETRY_GPU_LOCK_PATH = contract.DIRECT_GPU_SCHEDULER_LOCK_PATH
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_SCRIPT = Path(__file__).resolve().with_name("calibrate_p2_direct_soft_lag.py")
CALIBRATION_IMPLEMENTATION_PATH = CALIBRATION_SCRIPT.relative_to(REPOSITORY_ROOT).as_posix()
GPU_LOCK_IMPLEMENTATION_PATH = (
    Path(gpu_lock.__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
)
DEVICE = "cuda"
DTYPE = "bfloat16"
CALIBRATOR_PUBLICATION_SEMANTICS = "exclusive-atomic-mac-attested-full-external-binding"
CALIBRATOR_EXECUTION_SEMANTICS = "canonical-manifest-bound-sealed-memfd-v1"
CALIBRATOR_FD_ENV = "ADAPTIVE_V4_CANONICAL_CALIBRATOR_FD"
MATRIX_LOCK_SUFFIX = "p2-direct-calibration-matrix.lock"
MATRIX_LOCK_SEMANTICS = "persistent-sibling-flock-exclusive-process-owner-v1"
GPU_LEASE_SEMANTICS = "project-persistent-inode-exclusive-whole-matrix-v1"
GPU_LEASE_SCOPE = "before-preflight-through-terminal-validation"
GPU_LEASE_ACQUISITION_ORDER = "gpu-lease-before-matrix-lock-before-child-launch"
GPU_DEVICE_GUARD_SCOPE = "after-exact-environment-capture-through-terminal-validation"
CELL_CLAIM_NAME = ".p2-direct-calibration-cell.claim"
CELL_CLAIM_SEMANTICS = "exclusive-create-coordinate-nonce-preserve-on-failure-v2"
MATRIX_ATTESTATION_PURPOSE = "p2-direct-soft-lag-calibration-matrix-v1.2"
RETRY_ADMISSION_ATTESTATION_PURPOSE = "p2-direct-calibration-one-shot-retry-admission-v1.2"
CRASH_RECOVERY_BOUNDARY = (
    "fail-closed: a coordinate claim is preserved until a terminal artifact is validated; "
    "a surviving claim or an artifact outside the digest-bound completed matrix prefix is "
    "never auto-promoted and requires explicit operator quarantine"
)
CALIBRATOR_FD_BOOTSTRAP = (
    "import os,sys;"
    "p=sys.argv.pop(1);"
    "sys.path.insert(0,os.path.dirname(p));"
    f"fd=int(os.environ[{CALIBRATOR_FD_ENV!r}]);"
    "data=b'';"
    "\nwhile True:"
    "\n chunk=os.read(fd,1048576)"
    "\n if not chunk: break"
    "\n data+=chunk"
    "\ng={'__name__':'__main__','__file__':p,'__package__':None};"
    "exec(compile(data,p,'exec'),g,g)"
)

QUALITY_EVALUATION_STARTED = False
QUALITY_GATE = (
    "forbidden: this calibration runner never launches held-out quality evaluation, "
    "including after a terminal all-GO matrix"
)
CALIBRATION_MATRIX_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "terminal_decision",
        "source",
        "manifest",
        "attestation_contract",
        "scales",
        "frozen_training_seeds",
        "frozen_calibration_seeds",
        "coordinates",
        "device",
        "dtype",
        "calibration_script",
        "calibrator_publication_semantics",
        "calibrator_execution_semantics",
        "matrix_lock",
        "cell_claim_semantics",
        "gpu_lease",
        "execution_environment",
        "crash_recovery_boundary",
        "expected_cells",
        "completed_cells",
        "cell_decisions",
        "cells",
        "quality_evaluation_started",
        "quality_gate",
        "retry_admission",
        "payload_sha256",
        "attestation",
    }
)

RETRY_ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "admission_id",
        "status",
        "reason",
        "coordinate",
        "observed_terminal_decision",
        "superseded_manifest",
        "superseded_matrix_ledger",
        "preserved_claim",
        "preserved_calibration_artifact",
        "terminal_training_matrix_ledger",
        "checkpoint",
        "execution_environment",
        "current_manifest",
        "current_source",
        "incident_report",
        "quarantine_rule",
        "retry_rule",
        "scientific_subprocesses_started_at_creation",
        "payload_sha256",
        "attestation",
    }
)

RETRY_ADMISSION_BINDING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "bytes",
        "payload_sha256",
        "attestation_mac",
        "admission_id",
        "coordinate",
        "preserved_claim_sha256",
        "preserved_artifact_sha256",
    }
)
SUPERSEDED_CALIBRATION_MATRIX_FIELDS = CALIBRATION_MATRIX_FIELDS - {"retry_admission"}


@dataclass(frozen=True)
class CanonicalScriptSnapshot:
    path: str
    sha256: str
    bytes: int
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    @property
    def public_binding(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "implementation_path": CALIBRATION_IMPLEMENTATION_PATH,
        }


@dataclass(frozen=True)
class MatrixLayout:
    output_root: Path
    matrix_summary: Path
    training_output_root: Path
    lock_path: Path


@dataclass(frozen=True)
class ValidatedQuarantineEvidence:
    legacy_context: training_matrix.FrozenContext
    legacy_manifest_file_binding: dict[str, Any]
    matrix_ledger: dict[str, Any]
    matrix_ledger_binding: dict[str, Any]
    claim: dict[str, Any]
    claim_binding: dict[str, Any]
    artifact: dict[str, Any]
    artifact_binding: dict[str, Any]
    training_matrix_binding: dict[str, Any]
    checkpoint_binding: dict[str, Any]
    execution_environment: dict[str, Any]


@dataclass(frozen=True)
class ValidatedRetryAdmission:
    payload: dict[str, Any]
    public_binding: dict[str, Any]
    evidence: ValidatedQuarantineEvidence


@dataclass(frozen=True)
class _MatrixLockLease(Mapping[str, Any]):
    path: Path
    file_descriptor: int
    device: int
    inode: int
    owner: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.owner[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.owner)

    def __len__(self) -> int:
        return len(self.owner)

    def assert_held(self) -> None:
        try:
            opened = os.fstat(self.file_descriptor)
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                "Calibration matrix lock path was deleted or became inaccessible while held."
            ) from error
        _require(
            stat.S_ISREG(opened.st_mode),
            "Calibration matrix lock descriptor is no longer a regular file.",
        )
        _require(
            (opened.st_dev, opened.st_ino) == (self.device, self.inode),
            "Calibration matrix lock descriptor identity changed while held.",
        )
        _require(
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == (self.device, self.inode),
            "Calibration matrix lock path was replaced while held.",
        )
        _require(
            opened.st_uid == current.st_uid == os.getuid()
            and opened.st_nlink == current.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(current.st_mode) == 0o600,
            "Calibration matrix lock ownership, link count, or mode is unsafe.",
        )


_ACTIVE_MATRIX_LOCKS: set[str] = set()
_ACTIVE_MATRIX_LOCKS_GUARD = threading.Lock()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _exact_resolved_non_symlink_path(path: Path, *, label: str) -> Path:
    absolute = _absolute_path(path)
    _require(not absolute.is_symlink(), f"{label} may not be a symbolic link.")
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as error:
        raise ValueError(f"{label} cannot be resolved safely.") from error
    _require(
        resolved == absolute,
        f"{label} must use its exact resolved non-symlink path.",
    )
    return absolute


def _canonical_gpu_lock_path(path: Path) -> Path:
    absolute = _absolute_path(path)
    _require(not absolute.is_symlink(), "Calibration GPU lease path may not be a symbolic link.")
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as error:
        raise ValueError("Calibration GPU lease path cannot be resolved safely.") from error
    _require(
        resolved == absolute,
        "Calibration GPU lease path must use its exact resolved non-symlink path.",
    )
    return absolute


def _device_guard_binding(
    device_guard: gpu_lock.GPULockLease,
    *,
    routing_identity: Mapping[str, Any],
) -> dict[str, Any]:
    device_guard.assert_held()
    expected_path = gpu_lock.canonical_device_guard_path(routing_identity)
    _require(
        device_guard.path == expected_path,
        "Calibration physical-device guard path does not match the selected GPU.",
    )
    return {
        "path": str(expected_path),
        "semantics": gpu_lock.DEVICE_GUARD_SEMANTICS,
        "scope": GPU_DEVICE_GUARD_SCOPE,
        "implementation_path": GPU_LOCK_IMPLEMENTATION_PATH,
        "nonblocking": True,
        "persistent_inode": True,
        "device": device_guard.device,
        "inode": device_guard.inode,
    }


def _gpu_lease_binding(
    path: Path,
    *,
    frozen_execution_environment: Mapping[str, Any],
    device_guard: gpu_lock.GPULockLease,
) -> dict[str, Any]:
    routing_identity = execution_environment.selected_device_routing_identity(
        frozen_execution_environment
    )
    return {
        "path": str(_canonical_gpu_lock_path(path)),
        "semantics": GPU_LEASE_SEMANTICS,
        "scope": GPU_LEASE_SCOPE,
        "acquisition_order": GPU_LEASE_ACQUISITION_ORDER,
        "child_nested_lease": False,
        "portable_identity": "canonical-path-only-inode-excluded",
        "selected_device_class": execution_environment.selected_device_class(
            frozen_execution_environment
        ),
        "selected_device_routing_identity": routing_identity,
        "device_guard": _device_guard_binding(
            device_guard,
            routing_identity=routing_identity,
        ),
    }


def _validate_gpu_lease_binding(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "Calibration matrix GPU lease binding is missing.")
    raw = cast(Mapping[str, Any], value)
    _require(
        set(raw)
        == {
            "path",
            "semantics",
            "scope",
            "acquisition_order",
            "child_nested_lease",
            "portable_identity",
            "selected_device_class",
            "selected_device_routing_identity",
            "device_guard",
        }
        and isinstance(raw.get("path"), str)
        and bool(raw["path"]),
        "Calibration matrix GPU lease binding schema drifted.",
    )
    _require(
        raw.get("path") == str(_canonical_gpu_lock_path(Path(cast(str, raw["path"]))))
        and raw.get("semantics") == GPU_LEASE_SEMANTICS
        and raw.get("scope") == GPU_LEASE_SCOPE
        and raw.get("acquisition_order") == GPU_LEASE_ACQUISITION_ORDER
        and raw.get("child_nested_lease") is False
        and raw.get("portable_identity") == "canonical-path-only-inode-excluded",
        "Calibration matrix GPU lease semantics drifted.",
    )
    selected_class = raw.get("selected_device_class")
    routing_identity = raw.get("selected_device_routing_identity")
    guard = raw.get("device_guard")
    _require(
        isinstance(selected_class, Mapping)
        and set(selected_class) == execution_environment.SELECTED_DEVICE_CLASS_FIELDS
        and isinstance(routing_identity, Mapping)
        and set(routing_identity) == execution_environment.SELECTED_DEVICE_ROUTING_IDENTITY_FIELDS
        and isinstance(guard, Mapping),
        "Calibration matrix selected-device lease binding schema drifted.",
    )
    routing_identity = cast(Mapping[str, Any], routing_identity)
    guard = cast(Mapping[str, Any], guard)
    expected_guard_path = gpu_lock.canonical_device_guard_path(routing_identity)
    _require(
        set(guard)
        == {
            "path",
            "semantics",
            "scope",
            "implementation_path",
            "nonblocking",
            "persistent_inode",
            "device",
            "inode",
        }
        and guard.get("path") == str(expected_guard_path)
        and guard.get("semantics") == gpu_lock.DEVICE_GUARD_SEMANTICS
        and guard.get("scope") == GPU_DEVICE_GUARD_SCOPE
        and guard.get("implementation_path") == GPU_LOCK_IMPLEMENTATION_PATH
        and guard.get("nonblocking") is True
        and guard.get("persistent_inode") is True
        and type(guard.get("device")) is int
        and cast(int, guard["device"]) >= 0
        and type(guard.get("inode")) is int
        and cast(int, guard["inode"]) > 0,
        "Calibration matrix physical-device guard binding drifted.",
    )
    replayed = dict(raw)
    if expected is not None:
        _require(
            dict(raw) == dict(expected),
            "Calibration matrix GPU lease path or semantics drifted on resume.",
        )
    return replayed


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _matrix_lock_path(output_root: Path) -> Path:
    root = _absolute_path(output_root)
    _require(root != root.parent and bool(root.name), "Calibration output root may not be a root.")
    return root.parent / f".{root.name}.{MATRIX_LOCK_SUFFIX}"


def _validate_matrix_layout(
    *,
    output_root: Path,
    matrix_summary: Path,
    training_output_root: Path,
    attestation_key_path: Path | None = None,
) -> MatrixLayout:
    """Validate every mutable path before creating the persistent process lock."""

    root = _exact_resolved_non_symlink_path(output_root, label="Calibration output root")
    summary = _exact_resolved_non_symlink_path(matrix_summary, label="Calibration matrix summary")
    expected_summary = root / MATRIX_SUMMARY_NAME
    _require(
        summary == expected_summary,
        "Calibration matrix summary must be the canonical output-root summary path.",
    )
    _require(summary.parent == root and summary != root, "Calibration summary layout is invalid.")
    _require(
        not summary.exists() or summary.is_file(),
        "Calibration matrix summary is not a regular file.",
    )
    _require(
        not root.exists() or root.is_dir(),
        "Calibration output root is not a directory.",
    )

    training_root = _absolute_path(training_output_root).resolve(strict=False)
    superseded_root = _absolute_path(SUPERSEDED_OUTPUT_ROOT).resolve(strict=False)
    _require(
        not _paths_overlap(root, superseded_root)
        and not _paths_overlap(training_root, superseded_root),
        "Revision 1.2 writable/input roots may not overlap the immutable calibration quarantine.",
    )
    if REQUIRE_RETRY_ADMISSION:
        canonical_root = _absolute_path(OUTPUT_ROOT).resolve(strict=False)
        canonical_training_root = _absolute_path(TRAINING_OUTPUT_ROOT).resolve(strict=False)
        _require(
            root == canonical_root
            and summary == canonical_root / MATRIX_SUMMARY_NAME
            and training_root == canonical_training_root,
            "Revision 1.2 retry requires the canonical output, summary, and training roots.",
        )
    _require(
        not _paths_overlap(root, training_root),
        "Calibration and training output roots must be disjoint.",
    )

    lock_path = _exact_resolved_non_symlink_path(
        _matrix_lock_path(root), label="Calibration matrix lock"
    )
    _require(
        not _paths_overlap(lock_path, root),
        "Calibration matrix lock must be a sibling outside the output tree.",
    )
    _require(
        not _paths_overlap(lock_path, training_root),
        "Calibration matrix lock may not overlap the training input tree.",
    )
    _require(
        not _paths_overlap(lock_path, superseded_root),
        "Calibration matrix lock may not overlap the immutable calibration quarantine.",
    )
    _require(summary != lock_path, "Calibration summary and lock paths collided.")
    if attestation_key_path is not None:
        key_path = _absolute_path(attestation_key_path)
        _require(not key_path.is_symlink(), "Attestation key may not be a symbolic link.")
        key_path = key_path.resolve(strict=False)
        _require(
            not _paths_overlap(key_path, root)
            and not _paths_overlap(key_path, training_root)
            and key_path != lock_path,
            "Attestation key must be disjoint from output, training, and lock paths.",
        )
        _require(
            not key_path.is_relative_to(REPOSITORY_ROOT.resolve(strict=True)),
            "Attestation key must be outside the repository.",
        )

    reserved: set[Path] = {root, lock_path, training_root}
    for scale, training_seed, _, _ in _coordinates():
        scale_dir = root / scale
        output_dir = scale_dir / f"seed-{training_seed}"
        reserved.update(
            (
                scale_dir,
                output_dir,
                output_dir / f"{scale}-calibration.json",
                _absolute_path(
                    _training_summary_path(training_output_root, scale, training_seed)
                ).resolve(strict=False),
            )
        )
    _require(
        summary not in reserved,
        "Calibration matrix summary collides with a reserved output or training path.",
    )
    return MatrixLayout(
        output_root=root,
        matrix_summary=summary,
        training_output_root=training_output_root,
        lock_path=lock_path,
    )


def _write_lock_metadata(file_descriptor: int, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    os.ftruncate(file_descriptor, 0)
    offset = 0
    while offset < len(encoded):
        written = os.write(file_descriptor, encoded[offset:])
        _require(written > 0, "Calibration matrix lock metadata write stalled.")
        offset += written
    os.fsync(file_descriptor)


def _matrix_lock_binding(lock_path: Path) -> dict[str, Any]:
    return {
        "path": str(lock_path),
        "semantics": MATRIX_LOCK_SEMANTICS,
        "persistent_inode": True,
        "unlink_on_release": False,
        "kernel_releases_on_process_exit": True,
        "owner_metadata_advisory_after_crash": True,
    }


@contextmanager
def _exclusive_matrix_lock(lock_path: Path, *, matrix_summary: Path) -> Iterator[_MatrixLockLease]:
    """Hold one persistent-inode flock; never unlink it and split the lock domain.

    The kernel releases the lock on close or process death. Metadata is diagnostic: a SIGKILL can
    leave ``state=held`` even though the kernel lock is free, and the next owner overwrites it only
    after acquiring the flock. A process-local registry rejects nested/threaded reentry instead of
    allowing the same process to deadlock on a second open file description.
    """

    key = str(lock_path)
    with _ACTIVE_MATRIX_LOCKS_GUARD:
        _require(
            key not in _ACTIVE_MATRIX_LOCKS,
            "Calibration matrix lock is already held by this process; reentry is forbidden.",
        )
        _ACTIVE_MATRIX_LOCKS.add(key)

    file_descriptor: int | None = None
    acquired = False
    owner = {
        "schema_version": 1,
        "semantics": MATRIX_LOCK_SEMANTICS,
        "state": "held",
        "owner_nonce": secrets.token_hex(32),
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
        "acquired_time_ns": time.time_ns(),
        "matrix_summary": str(matrix_summary),
    }
    outcome = "released"
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        no_follow = getattr(os, "O_NOFOLLOW", None)
        _require(no_follow is not None, "Calibration matrix locking requires O_NOFOLLOW support.")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
        try:
            file_descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise ValueError("Calibration matrix lock could not be opened safely.") from error
        opened = os.fstat(file_descriptor)
        current = os.stat(lock_path, follow_symlinks=False)
        _require(stat.S_ISREG(opened.st_mode), "Calibration matrix lock is not a regular file.")
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Calibration matrix lock path changed while it was opened.",
        )
        _require(
            opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o600,
            "Calibration matrix lock ownership, link count, or mode is unsafe.",
        )
        lease = _MatrixLockLease(
            path=lock_path,
            file_descriptor=file_descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
            owner=owner,
        )
        lease.assert_held()
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        acquired = True
        lease.assert_held()
        _write_lock_metadata(file_descriptor, owner)
        lease.assert_held()
        try:
            yield lease
        except BaseException:
            outcome = "released-after-error"
            raise
        finally:
            lease.assert_held()
            released = {
                **owner,
                "state": outcome,
                "released_time_ns": time.time_ns(),
            }
            try:
                lease.assert_held()
                _write_lock_metadata(file_descriptor, released)
                lease.assert_held()
            finally:
                fcntl.flock(file_descriptor, fcntl.LOCK_UN)
                acquired = False
    finally:
        if file_descriptor is not None:
            if acquired:
                fcntl.flock(file_descriptor, fcntl.LOCK_UN)
            os.close(file_descriptor)
        with _ACTIVE_MATRIX_LOCKS_GUARD:
            _ACTIVE_MATRIX_LOCKS.discard(key)


def _assert_cell_claim_identity(
    descriptor: int,
    claim_path: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> None:
    opened = os.fstat(descriptor)
    try:
        current = os.stat(claim_path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("Calibration cell claim disappeared while held.") from error
    _require(
        (opened.st_dev, opened.st_ino)
        == (current.st_dev, current.st_ino)
        == (expected_device, expected_inode),
        "Calibration cell claim path was replaced while held.",
    )
    _require(
        stat.S_ISREG(opened.st_mode)
        and stat.S_ISREG(current.st_mode)
        and opened.st_uid == current.st_uid == os.getuid()
        and opened.st_nlink == current.st_nlink == 1
        and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(current.st_mode) == 0o600,
        "Calibration cell claim ownership, links, or mode changed while held.",
    )


@contextmanager
def _exclusive_cell_claim(
    output_dir: Path,
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    evaluation_seed: int,
    launch_nonce: str,
) -> Iterator[dict[str, Any]]:
    _durable_mkdir(output_dir.parent)
    _durable_mkdir(output_dir)
    claim_path = output_dir / CELL_CLAIM_NAME
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Calibration cell claims require O_NOFOLLOW.")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
    )
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(
            f"Calibration cell has an orphan or active claim: {scale}/{training_seed}."
        ) from error
    metadata = os.fstat(descriptor)
    release_claim = False
    try:
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600,
            "Calibration cell claim metadata is unsafe.",
        )
        _write_lock_metadata(
            descriptor,
            {
                "schema_version": 1,
                "semantics": CELL_CLAIM_SEMANTICS,
                "coordinate": {
                    "scale": scale,
                    "training_seed": training_seed,
                    "calibration_seed": calibration_seed,
                    "evaluation_seed_reserved": evaluation_seed,
                },
                "launch_nonce": launch_nonce,
                "pid": os.getpid(),
                "created_time_ns": time.time_ns(),
            },
        )
        _fsync_directory(output_dir)
        _assert_cell_claim_identity(
            descriptor,
            claim_path,
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )
        yield {
            "path": str(claim_path),
            "semantics": CELL_CLAIM_SEMANTICS,
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
        }
        release_claim = True
    finally:
        try:
            _assert_cell_claim_identity(
                descriptor,
                claim_path,
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
            )
            if release_claim:
                claim_path.unlink()
                _fsync_directory(output_dir)
                _require(
                    os.fstat(descriptor).st_nlink == 0 and not os.path.lexists(claim_path),
                    "Calibration cell claim release did not remove the held path.",
                )
        finally:
            os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_fd(file_descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(file_descriptor, 1024 * 1024, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _canonical_calibration_script(candidate: Path) -> Path:
    _require(not candidate.is_symlink(), "Calibration script may not be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
        canonical = CALIBRATION_SCRIPT.resolve(strict=True)
    except OSError as error:
        raise ValueError("Canonical calibration script is missing or inaccessible.") from error
    _require(
        resolved == canonical,
        "Calibration script must resolve to the manifest-bound canonical calibrator.",
    )
    _require(not canonical.is_symlink(), "Canonical calibration script may not be a symbolic link.")
    return canonical


def _assert_calibration_implementation_binding(
    context: training_matrix.FrozenContext,
) -> None:
    _require(
        CALIBRATION_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS,
        "Canonical calibrator is absent from the frozen implementation inventory.",
    )
    _require(
        contract.implementation_tree_digest() == context.manifest_binding["implementation_digest"],
        "Canonical calibrator is not bound by the frozen implementation manifest.",
    )


def _open_canonical_script(
    canonical: Path,
    *,
    expected: CanonicalScriptSnapshot | None = None,
) -> tuple[int, CanonicalScriptSnapshot]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Stable calibrator execution requires O_NOFOLLOW support.")
    no_follow = cast(int, no_follow)
    try:
        file_descriptor = os.open(canonical, flags | no_follow)
    except OSError as error:
        raise ValueError("Canonical calibration script could not be opened safely.") from error
    try:
        opened = os.fstat(file_descriptor)
        current = os.stat(canonical, follow_symlinks=False)
        _require(stat.S_ISREG(opened.st_mode), "Canonical calibration script is not regular.")
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Canonical calibration script changed while it was opened.",
        )
        snapshot = CanonicalScriptSnapshot(
            path=str(canonical),
            sha256=_sha256_fd(file_descriptor),
            bytes=opened.st_size,
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=opened.st_mode,
            mtime_ns=opened.st_mtime_ns,
            ctime_ns=opened.st_ctime_ns,
        )
        if expected is not None:
            _require(
                snapshot == expected,
                "Canonical calibration script changed after frozen preflight.",
            )
        return file_descriptor, snapshot
    except BaseException:
        os.close(file_descriptor)
        raise


def _validate_open_script(
    file_descriptor: int,
    *,
    expected: CanonicalScriptSnapshot,
) -> None:
    opened = os.fstat(file_descriptor)
    current = os.stat(expected.path, follow_symlinks=False)
    observed = CanonicalScriptSnapshot(
        path=expected.path,
        sha256=_sha256_fd(file_descriptor),
        bytes=opened.st_size,
        device=opened.st_dev,
        inode=opened.st_ino,
        mode=opened.st_mode,
        mtime_ns=opened.st_mtime_ns,
        ctime_ns=opened.st_ctime_ns,
    )
    _require(observed == expected, "Open canonical calibration script changed during execution.")
    _require(
        (current.st_dev, current.st_ino) == (expected.device, expected.inode),
        "Canonical calibration path changed during execution.",
    )


def _sealed_script_copy(
    file_descriptor: int,
    *,
    expected: CanonicalScriptSnapshot,
) -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(
        create is not None and allow_sealing is not None,
        "Stable calibrator execution requires sealed memfd support.",
    )
    create = cast(Callable[[str, int], int], create)
    allow_sealing = cast(int, allow_sealing)
    sealed = create(
        "adaptive-v4-canonical-calibrator",
        cast(int, getattr(os, "MFD_CLOEXEC", 0)) | allow_sealing,
    )
    try:
        offset = 0
        while offset < expected.bytes:
            chunk = os.pread(file_descriptor, min(1024 * 1024, expected.bytes - offset), offset)
            _require(bool(chunk), "Canonical calibrator became truncated during snapshotting.")
            written = 0
            while written < len(chunk):
                written += os.write(sealed, chunk[written:])
            offset += len(chunk)
        _require(
            _sha256_fd(sealed) == expected.sha256,
            "Sealed calibrator snapshot digest drifted.",
        )
        required_seals = (
            fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(sealed, fcntl.F_ADD_SEALS, required_seals)
        _require(
            fcntl.fcntl(sealed, fcntl.F_GET_SEALS) & required_seals == required_seals,
            "Calibrator memfd snapshot was not sealed.",
        )
        os.lseek(sealed, 0, os.SEEK_SET)
        return sealed
    except BaseException:
        os.close(sealed)
        raise


def _run_calibrator_from_stable_script(
    command: Sequence[str],
    *,
    canonical: Path,
    expected: CanonicalScriptSnapshot,
    trust_root: attestation.TrustRoot,
    gpu_lease: gpu_lock.GPULockLease,
    device_guard: gpu_lock.GPULockLease,
) -> subprocess.CompletedProcess[Any]:
    gpu_lease.assert_held()
    device_guard.assert_held()
    file_descriptor, snapshot = _open_canonical_script(canonical, expected=expected)
    sealed_descriptor: int | None = None
    key_descriptor: int | None = None
    try:
        sealed_descriptor = _sealed_script_copy(file_descriptor, expected=snapshot)
        key_descriptor = attestation.create_sealed_key_fd(trust_root)
        environment = os.environ.copy()
        environment.pop(attestation.KEY_PATH_ENV, None)
        environment[CALIBRATOR_FD_ENV] = str(sealed_descriptor)
        environment[attestation.KEY_FD_ENV] = str(key_descriptor)
        result = subprocess.run(
            list(command),
            check=False,
            pass_fds=(
                sealed_descriptor,
                key_descriptor,
                gpu_lease.fileno(),
                device_guard.fileno(),
            ),
            env=environment,
        )
        gpu_lease.assert_held()
        device_guard.assert_held()
        _validate_open_script(file_descriptor, expected=snapshot)
        return result
    finally:
        if key_descriptor is not None:
            os.close(key_descriptor)
        if sealed_descriptor is not None:
            os.close(sealed_descriptor)
        os.close(file_descriptor)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label} JSON: {path}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
    return payload


def _digest_bound_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    digest_bound = dict(payload)
    digest_bound.pop("attestation", None)
    digest_bound.pop("payload_sha256", None)
    digest_bound["payload_sha256"] = contract.json_digest(digest_bound)
    return digest_bound


def _attested_payload(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    purpose: str = MATRIX_ATTESTATION_PURPOSE,
) -> dict[str, Any]:
    digest_bound = _digest_bound_payload(payload)
    digest_bound["attestation"] = attestation.attest_payload(
        digest_bound,
        trust_root=trust_root,
        purpose=purpose,
    )
    return digest_bound


def _validate_payload_digest(payload: Mapping[str, Any]) -> None:
    digest = payload.get("payload_sha256")
    _require(contract.is_sha256(digest), "Calibration matrix payload digest is invalid.")
    digest_source = dict(payload)
    digest_source.pop("attestation", None)
    digest_source.pop("payload_sha256", None)
    _require(
        digest == contract.json_digest(digest_source),
        "Calibration matrix payload digest does not match its contents.",
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_matrix_ledger(
    path: Path,
    payload: Mapping[str, Any],
    *,
    matrix_lock: _MatrixLockLease,
    trust_root: attestation.TrustRoot,
) -> None:
    matrix_lock.assert_held()
    _atomic_write_json(path, payload)
    matrix_lock.assert_held()
    opened = attestation.open_regular_nofollow(path)
    try:
        metadata = os.fstat(opened.file_descriptor)
        _require(
            metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600,
            "Published calibration matrix ownership, link count, or mode is unsafe.",
        )
        raw = opened.read_bytes()
        expected = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        _require(raw == expected, "Published calibration matrix bytes drifted.")
        _validate_payload_digest(payload)
        envelope = payload.get("attestation")
        _require(isinstance(envelope, Mapping), "Published matrix attestation is missing.")
        semantic = dict(payload)
        semantic.pop("attestation")
        attestation.verify_attestation(
            semantic,
            cast(Mapping[str, Any], envelope),
            trust_root=trust_root,
            purpose=MATRIX_ATTESTATION_PURPOSE,
        )
        opened.assert_unchanged()
    finally:
        opened.close()


def _coordinates() -> tuple[tuple[str, int, int, int], ...]:
    coordinates: list[tuple[str, int, int, int]] = []
    calibration_seeds: list[int] = []
    for scale in FROZEN_SCALES:
        for training_seed in FROZEN_TRAINING_SEEDS:
            aligned_training, calibration_seed, evaluation_seed = contract.seed_triplet(
                training_seed
            )
            _require(aligned_training == training_seed, "Training seed alignment drifted.")
            _require(
                calibration_seed in FROZEN_CALIBRATION_SEEDS,
                "Calibration seed alignment drifted.",
            )
            coordinates.append((scale, training_seed, calibration_seed, evaluation_seed))
            if scale == FROZEN_SCALES[0]:
                calibration_seeds.append(calibration_seed)
    _require(
        tuple(calibration_seeds) == FROZEN_CALIBRATION_SEEDS,
        "Frozen calibration seed inventory drifted.",
    )
    _require(len(coordinates) == EXPECTED_CELLS, "Frozen calibration grid size drifted.")
    return tuple(coordinates)


def _cell_output_dir(output_root: Path, scale: str, training_seed: int) -> Path:
    return output_root / scale / f"seed-{training_seed}"


def _artifact_path(output_root: Path, scale: str, training_seed: int) -> Path:
    return _cell_output_dir(output_root, scale, training_seed) / f"{scale}-calibration.json"


def _training_summary_path(training_output_root: Path, scale: str, training_seed: int) -> Path:
    return training_output_root / scale / f"seed-{training_seed}" / f"{scale}-training.summary.json"


def _validate_training_input(
    *,
    training_output_root: Path,
    scale: str,
    training_seed: int,
    context: training_matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
    trainer_binding: Mapping[str, Any],
    ledger_record: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    summary_path = _training_summary_path(training_output_root, scale, training_seed)
    _require(
        summary_path.is_file(), f"Direct training summary is missing: {scale}/{training_seed}."
    )
    summary, checkpoint, _raw_checkpoint = (
        training_matrix.load_validated_training_bundle_for_ledger_record(
            summary_path,
            output_root=training_output_root,
            context=context,
            scale=scale,
            seed=training_seed,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
            ledger_record=ledger_record,
        )
    )
    _require(
        ledger_record.get("checkpoint") == checkpoint,
        "Terminal training ledger checkpoint binding drifted.",
    )
    return summary_path, summary, checkpoint


def build_calibration_command(
    *,
    calibration_script: Path,
    artifact_path: Path,
    manifest_path: Path,
    training_manifest_path: Path | None = None,
    training_summary_path: Path,
    training_matrix_summary_path: Path,
    checkpoint_path: Path,
    scale: str,
    training_seed: int,
    device_index: int = 0,
    device_routing_identity: Mapping[str, Any],
) -> list[str]:
    _require(scale in FROZEN_SCALES, "Calibration command scale is not frozen.")
    _require(training_seed in FROZEN_TRAINING_SEEDS, "Calibration command seed is not frozen.")
    _require(
        type(device_index) is int and device_index >= 0,
        "Calibration command device index is invalid.",
    )
    routing_identity = execution_environment.validate_device_routing_identity(
        device_routing_identity
    )
    canonical = _canonical_calibration_script(calibration_script)
    return [
        sys.executable,
        "-I",
        "-c",
        CALIBRATOR_FD_BOOTSTRAP,
        str(canonical),
        "--checkpoint",
        str(checkpoint_path.resolve()),
        "--scale",
        scale,
        "--training-seed",
        str(training_seed),
        "--output",
        str(artifact_path),
        "--manifest",
        str(manifest_path.resolve()),
        "--training-manifest",
        str(
            (manifest_path if training_manifest_path is None else training_manifest_path).resolve()
        ),
        "--training-summary",
        str(training_summary_path.resolve()),
        "--training-matrix-summary",
        str(training_matrix_summary_path.resolve()),
        "--device",
        f"cuda:{device_index}",
        "--expected-device-routing-identity-json",
        json.dumps(routing_identity, sort_keys=True, separators=(",", ":")),
        "--dtype",
        DTYPE,
    ]


def _expected_training_binding(
    *,
    summary_path: Path,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    scale: str,
    training_seed: int,
    training_matrix_summary_path: Path,
    training_matrix_payload: Mapping[str, Any],
    ledger_record: Mapping[str, Any],
    result_context: training_matrix.FrozenContext,
    training_context: training_matrix.FrozenContext,
) -> dict[str, Any]:
    binding = {
        "path": str(summary_path.resolve()),
        "sha256": _sha256(summary_path),
        "bytes": summary_path.stat().st_size,
        "payload_sha256": summary.get("payload_sha256"),
        "attestation_mac": summary["attestation"]["mac"],
        "experiment_id": summary.get("experiment_id"),
        "scale": scale,
        "training_seed": training_seed,
        "checkpoint_sha256": checkpoint.get("sha256"),
        "launch_nonce": ledger_record.get("launch_nonce"),
        "canonical_trainer_sha256": ledger_record.get("canonical_trainer_sha256"),
        "terminal_matrix_ledger": {
            "path": str(training_matrix_summary_path.resolve()),
            "sha256": _sha256(training_matrix_summary_path),
            "bytes": training_matrix_summary_path.stat().st_size,
            "payload_sha256": training_matrix_payload.get("payload_sha256"),
            "attestation_mac": training_matrix_payload["attestation"]["mac"],
            "status": training_matrix_payload.get("status"),
        },
    }
    if training_context.manifest_binding != result_context.manifest_binding:
        binding["training_manifest"] = training_context.manifest_binding
    return binding


def _expected_checkpoint_binding(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    path = checkpoint.get("path")
    if not isinstance(path, str):
        raise ValueError("Validated checkpoint path is invalid.")
    return {
        # Preserve the exact authenticated upstream spelling.  Referent,
        # digest, size, and no-symlink checks are performed separately by the
        # terminal training-bundle validator and the calibrator CLI boundary.
        "path": path,
        "sha256": checkpoint.get("sha256"),
        "bytes": checkpoint.get("bytes"),
    }


def _expected_exit_code(decision: str) -> int:
    _require(decision in {"GO", "NO-GO"}, "Calibration decision must be explicit GO or NO-GO.")
    return 0 if decision == "GO" else 2


def load_and_validate_calibration_artifact(
    artifact_path: Path,
    *,
    scale: str,
    training_seed: int,
    calibration_seed: int,
    evaluation_seed: int,
    context: training_matrix.FrozenContext,
    training_context: training_matrix.FrozenContext,
    summary_path: Path,
    summary: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    command: Sequence[str],
    trust_root: attestation.TrustRoot,
    training_matrix_summary_path: Path,
    training_matrix_payload: Mapping[str, Any],
    ledger_record: Mapping[str, Any],
) -> dict[str, Any]:
    _require(artifact_path.is_file(), f"Calibration artifact is missing: {artifact_path}")
    _require(not artifact_path.is_symlink(), "Calibration artifact may not be a symbolic link.")
    payload = _load_json(artifact_path, label="calibration artifact")
    calibration.validate_calibration_artifact(
        payload,
        verify_bindings=True,
        trust_root=trust_root,
    )
    _require(payload.get("scale") == scale, "Calibration artifact scale drifted.")
    _require(payload.get("training_seed") == training_seed, "Calibration training seed drifted.")
    _require(
        payload.get("seed") == payload.get("calibration_seed") == calibration_seed,
        "Calibration seed coordinate drifted.",
    )
    _require(
        payload.get("evaluation_seed_reserved") == evaluation_seed,
        "Reserved evaluation seed drifted.",
    )
    _require(payload.get("source") == context.source, "Calibration source binding drifted.")
    _require(
        payload.get("manifest") == context.manifest_binding,
        "Calibration manifest binding drifted.",
    )
    _require(
        payload.get("checkpoint") == _expected_checkpoint_binding(checkpoint),
        "Calibration checkpoint binding drifted.",
    )
    _require(
        payload.get("training_summary")
        == _expected_training_binding(
            summary_path=summary_path,
            summary=summary,
            checkpoint=checkpoint,
            scale=scale,
            training_seed=training_seed,
            training_matrix_summary_path=training_matrix_summary_path,
            training_matrix_payload=training_matrix_payload,
            ledger_record=ledger_record,
            result_context=context,
            training_context=training_context,
        ),
        "Calibration training-summary binding drifted.",
    )
    decision = payload.get("terminal_decision")
    if not isinstance(decision, str):
        raise ValueError("Calibration terminal decision is missing.")
    _expected_exit_code(decision)
    budget_decisions = payload.get("budget_decisions")
    if not isinstance(budget_decisions, Mapping):
        raise ValueError("Calibration budget decisions are missing.")
    _require(
        set(budget_decisions) == set(calibration.FROZEN_BUDGETS)
        and all(value in {"GO", "NO-GO"} for value in budget_decisions.values()),
        "Calibration budget decisions are not explicit GO/NO-GO evidence.",
    )
    expected_decision = (
        "GO" if all(value == "GO" for value in budget_decisions.values()) else "NO-GO"
    )
    _require(decision == expected_decision, "Calibration aggregate decision drifted.")
    return payload


def _cell_record(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    command: Sequence[str],
) -> dict[str, Any]:
    decision = cast(str, payload["terminal_decision"])
    return {
        "scale": payload["scale"],
        "training_seed": payload["training_seed"],
        "calibration_seed": payload["calibration_seed"],
        "evaluation_seed_reserved": payload["evaluation_seed_reserved"],
        "status": "terminal",
        "terminal_decision": decision,
        "budget_decisions": payload["budget_decisions"],
        "calibrator_exit_code": _expected_exit_code(decision),
        "command": list(command),
        "checkpoint": payload["checkpoint"],
        "training_summary": payload["training_summary"],
        "artifact": {
            "path": str(artifact_path),
            "sha256": _sha256(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "payload_sha256": payload["payload_sha256"],
        },
    }


def _matrix_payload(
    cells: Sequence[Mapping[str, Any]],
    *,
    context: training_matrix.FrozenContext,
    calibration_script_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    gpu_lease_binding: Mapping[str, Any],
    execution_environment_binding: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
    retry_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coordinates = _coordinates()
    terminal = len(cells) == len(coordinates)
    decisions = {
        f"{cell['scale']}/{cell['training_seed']}/{cell['calibration_seed']}": cell[
            "terminal_decision"
        ]
        for cell in cells
    }
    terminal_decision: str | None = None
    if terminal:
        terminal_decision = "GO" if all(value == "GO" for value in decisions.values()) else "NO-GO"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal" if terminal else "in_progress",
        "terminal_decision": terminal_decision,
        "source": context.source,
        "manifest": context.manifest_binding,
        "attestation_contract": context.manifest_binding["attestation"],
        "scales": list(FROZEN_SCALES),
        "frozen_training_seeds": list(FROZEN_TRAINING_SEEDS),
        "frozen_calibration_seeds": list(FROZEN_CALIBRATION_SEEDS),
        "coordinates": [
            {
                "scale": scale,
                "training_seed": training_seed,
                "calibration_seed": calibration_seed,
                "evaluation_seed_reserved": evaluation_seed,
            }
            for scale, training_seed, calibration_seed, evaluation_seed in coordinates
        ],
        "device": execution_environment.explicit_cuda_device_spec(execution_environment_binding),
        "dtype": DTYPE,
        "calibration_script": dict(calibration_script_binding),
        "calibrator_publication_semantics": CALIBRATOR_PUBLICATION_SEMANTICS,
        "calibrator_execution_semantics": CALIBRATOR_EXECUTION_SEMANTICS,
        "matrix_lock": dict(matrix_lock_binding),
        "cell_claim_semantics": CELL_CLAIM_SEMANTICS,
        "gpu_lease": dict(gpu_lease_binding),
        "execution_environment": execution_environment.validate_execution_environment(
            execution_environment_binding
        ),
        "crash_recovery_boundary": CRASH_RECOVERY_BOUNDARY,
        "expected_cells": len(coordinates),
        "completed_cells": len(cells),
        "cell_decisions": decisions,
        "cells": list(cells),
        "quality_evaluation_started": QUALITY_EVALUATION_STARTED,
        "quality_gate": QUALITY_GATE,
        "retry_admission": None if retry_admission is None else dict(retry_admission),
    }
    return _attested_payload(payload, trust_root=trust_root)


def validate_matrix_summary(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    training_output_root: Path,
    calibration_script: Path,
    calibration_script_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    expected_gpu_lease: Mapping[str, Any] | None = None,
    expected_execution_environment: Mapping[str, Any] | None = None,
    context: training_matrix.FrozenContext,
    training_context: training_matrix.FrozenContext | None = None,
    trust_root: attestation.TrustRoot,
    training_matrix_summary_path: Path,
    training_matrix_payload: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    ledger_records: Mapping[tuple[str, int], Mapping[str, Any]],
    retry_admission: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    validated_training_context = context if training_context is None else training_context
    _require(
        set(payload) == CALIBRATION_MATRIX_FIELDS,
        "Calibration matrix top-level schema drifted.",
    )
    _validate_payload_digest(payload)
    raw_envelope = payload.get("attestation")
    _require(isinstance(raw_envelope, Mapping), "Calibration matrix attestation is missing.")
    envelope = cast(Mapping[str, Any], raw_envelope)
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        envelope,
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
    )
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Calibration matrix schema drifted.")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong calibration matrix ID.")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "Calibration matrix type drifted.")
    _require(payload.get("source") == context.source, "Calibration matrix source drifted.")
    _require(payload.get("manifest") == context.manifest_binding, "Matrix manifest drifted.")
    _require(
        payload.get("attestation_contract") == context.manifest_binding["attestation"],
        "Calibration matrix attestation contract drifted.",
    )
    _require(tuple(payload.get("scales", ())) == FROZEN_SCALES, "Matrix scales drifted.")
    _require(
        tuple(payload.get("frozen_training_seeds", ())) == FROZEN_TRAINING_SEEDS,
        "Matrix training seeds drifted.",
    )
    _require(
        tuple(payload.get("frozen_calibration_seeds", ())) == FROZEN_CALIBRATION_SEEDS,
        "Matrix calibration seeds drifted.",
    )
    coordinates = _coordinates()
    expected_coordinate_payload = [
        {
            "scale": scale,
            "training_seed": training_seed,
            "calibration_seed": calibration_seed,
            "evaluation_seed_reserved": evaluation_seed,
        }
        for scale, training_seed, calibration_seed, evaluation_seed in coordinates
    ]
    _require(
        payload.get("coordinates") == expected_coordinate_payload,
        "Calibration coordinate inventory drifted.",
    )
    _require(payload.get("dtype") == DTYPE, "Calibration matrix is not bfloat16-bound.")
    _require(
        payload.get("calibration_script") == dict(calibration_script_binding),
        "Calibration matrix canonical-script binding drifted.",
    )
    _require(
        payload.get("calibrator_publication_semantics") == CALIBRATOR_PUBLICATION_SEMANTICS,
        "Calibration publication semantics drifted.",
    )
    _require(
        payload.get("calibrator_execution_semantics") == CALIBRATOR_EXECUTION_SEMANTICS,
        "Calibration stable-execution semantics drifted.",
    )
    _require(
        payload.get("matrix_lock") == dict(matrix_lock_binding),
        "Calibration matrix process-lock binding drifted.",
    )
    _require(
        payload.get("cell_claim_semantics") == CELL_CLAIM_SEMANTICS,
        "Calibration cell-claim semantics drifted.",
    )
    validated_gpu_lease = _validate_gpu_lease_binding(
        payload.get("gpu_lease"),
        expected=expected_gpu_lease,
    )
    raw_environment = payload.get("execution_environment")
    _require(
        isinstance(raw_environment, Mapping),
        "Calibration matrix execution environment is missing.",
    )
    frozen_environment = execution_environment.validate_execution_environment(
        cast(Mapping[str, Any], raw_environment)
    )
    _require(
        payload.get("device")
        == execution_environment.explicit_cuda_device_spec(frozen_environment),
        "Calibration matrix logical CUDA route drifted.",
    )
    _require(
        validated_gpu_lease["selected_device_class"]
        == execution_environment.selected_device_class(frozen_environment)
        and validated_gpu_lease["selected_device_routing_identity"]
        == execution_environment.selected_device_routing_identity(frozen_environment),
        "Calibration matrix GPU lease is not bound to its exact selected device.",
    )
    if expected_execution_environment is not None:
        _require(
            frozen_environment
            == execution_environment.validate_execution_environment(expected_execution_environment),
            "Calibration matrix execution environment changed on exact resume.",
        )
    _require(
        payload.get("crash_recovery_boundary") == CRASH_RECOVERY_BOUNDARY,
        "Calibration crash-recovery boundary drifted.",
    )
    _require(payload.get("expected_cells") == len(coordinates), "Expected cell count drifted.")
    cells = payload.get("cells")
    _require(isinstance(cells, list), "Calibration matrix cell inventory is invalid.")
    cells = cast(list[Mapping[str, Any]], cells)
    _require(payload.get("completed_cells") == len(cells), "Completed cell count drifted.")
    _require(len(cells) <= len(coordinates), "Calibration matrix contains extra cells.")
    terminal = len(cells) == len(coordinates)
    _require(
        payload.get("status") == ("terminal" if terminal else "in_progress"),
        "Calibration matrix status drifted.",
    )
    _require(
        payload.get("quality_evaluation_started") is False
        and payload.get("quality_gate") == QUALITY_GATE,
        "Calibration runner may not start quality evaluation.",
    )
    _require(
        payload.get("retry_admission")
        == (None if retry_admission is None else dict(retry_admission)),
        "Calibration retry-admission binding drifted.",
    )
    if retry_admission is not None:
        _assert_retry_admission_file_binding(retry_admission)

    validated: list[dict[str, Any]] = []
    decisions: dict[str, str] = {}
    for stored_record, coordinate in zip(cells, coordinates[: len(cells)], strict=True):
        scale, training_seed, calibration_seed, evaluation_seed = coordinate
        summary_path, summary, checkpoint = _validate_training_input(
            training_output_root=training_output_root,
            scale=scale,
            training_seed=training_seed,
            context=validated_training_context,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
            ledger_record=ledger_records[(scale, training_seed)],
        )
        artifact_path = _artifact_path(output_root, scale, training_seed)
        checkpoint_path = Path(cast(str, checkpoint["path"]))
        command = build_calibration_command(
            calibration_script=calibration_script,
            artifact_path=artifact_path,
            manifest_path=context.manifest_path,
            training_manifest_path=validated_training_context.manifest_path,
            training_summary_path=summary_path,
            training_matrix_summary_path=training_matrix_summary_path,
            checkpoint_path=checkpoint_path,
            scale=scale,
            training_seed=training_seed,
            device_index=cast(int, frozen_environment["current_device_index"]),
            device_routing_identity=(
                execution_environment.selected_device_routing_identity(frozen_environment)
            ),
        )
        artifact = load_and_validate_calibration_artifact(
            artifact_path,
            scale=scale,
            training_seed=training_seed,
            calibration_seed=calibration_seed,
            evaluation_seed=evaluation_seed,
            context=context,
            training_context=validated_training_context,
            summary_path=summary_path,
            summary=summary,
            checkpoint=checkpoint,
            command=command,
            trust_root=trust_root,
            training_matrix_summary_path=training_matrix_summary_path,
            training_matrix_payload=training_matrix_payload,
            ledger_record=ledger_records[(scale, training_seed)],
        )
        _require(
            artifact.get("environment") == frozen_environment,
            f"Calibration execution environment drifted: {scale}/{training_seed}.",
        )
        expected_record = _cell_record(artifact, artifact_path=artifact_path, command=command)
        _require(
            stored_record == expected_record,
            f"Calibration matrix cell record drifted: {scale}/{training_seed}.",
        )
        validated.append(expected_record)
        decisions[f"{scale}/{training_seed}/{calibration_seed}"] = cast(
            str, artifact["terminal_decision"]
        )

    _require(payload.get("cell_decisions") == decisions, "Matrix cell decisions drifted.")
    expected_terminal_decision: str | None = None
    if terminal:
        expected_terminal_decision = (
            "GO" if all(value == "GO" for value in decisions.values()) else "NO-GO"
        )
    _require(
        payload.get("terminal_decision") == expected_terminal_decision,
        "Calibration matrix terminal decision drifted.",
    )
    matrix_summary_path = Path(os.path.abspath(output_root)) / MATRIX_SUMMARY_NAME
    opened_matrix = attestation.open_regular_nofollow(matrix_summary_path)
    try:
        metadata = os.fstat(opened_matrix.file_descriptor)
        _require(
            metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600,
            "Calibration matrix ledger ownership, link count, or mode is unsafe.",
        )
        expected_matrix_bytes = (
            json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        _require(
            opened_matrix.read_bytes() == expected_matrix_bytes,
            "Calibration matrix ledger is not the exact canonical published byte encoding.",
        )
        opened_matrix.assert_unchanged()
    finally:
        opened_matrix.close()
    _preflight_output_tree(
        output_root=output_root,
        matrix_summary=matrix_summary_path,
        completed_cells=len(validated),
        retry_admission_path=(
            None
            if retry_admission is None
            else Path(cast(str, retry_admission["path"]))
        ),
    )
    return validated


def _assert_empty_cell_output(output_dir: Path) -> None:
    if output_dir.exists():
        _require(
            not os.path.lexists(output_dir / CELL_CLAIM_NAME),
            f"Refusing orphan calibration cell claim: {output_dir}",
        )
        _require(
            not any(output_dir.iterdir()),
            f"Refusing orphaned or stale calibration output: {output_dir}",
        )


def _assert_frozen_contract() -> None:
    _require(tuple(contract.SCALES) == FROZEN_SCALES, "Contract calibration scales drifted.")
    _require(
        tuple(contract.TRAINING_SEEDS) == FROZEN_TRAINING_SEEDS,
        "Contract training seeds drifted.",
    )
    _require(
        tuple(training_matrix.FROZEN_SCALES) == FROZEN_SCALES
        and tuple(calibration.FROZEN_SCALES) == FROZEN_SCALES,
        "Training/calibration implementation scales drifted.",
    )
    _require(
        tuple(training_matrix.FROZEN_TRAINING_SEEDS) == FROZEN_TRAINING_SEEDS
        and tuple(calibration.FROZEN_TRAINING_SEEDS) == FROZEN_TRAINING_SEEDS,
        "Training/calibration implementation seeds drifted.",
    )
    _require(
        tuple(calibration.FROZEN_CALIBRATION_SEEDS) == FROZEN_CALIBRATION_SEEDS,
        "Calibration implementation seed grid drifted.",
    )


def _v1_1_manifest_path() -> Path:
    candidate = Path(contract.V1_1_MANIFEST_PATH)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _load_v1_1_context(
    *, trust_root: attestation.TrustRoot
) -> training_matrix.FrozenContext:
    """Load the exact revision 1.1 context for immutable prerequisite validation."""

    context = training_matrix.load_v1_1_frozen_context(trust_root=trust_root)
    _require(
        context.manifest_path == _v1_1_manifest_path().resolve(),
        "Revision 1.1 manifest path drifted from the retry amendment.",
    )
    return context


def _assert_safe_quarantine_file(
    opened: attestation.OpenedRegularFile,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> None:
    file_stat = os.fstat(opened.file_descriptor)
    _require(
        opened.sha256 == expected_sha256 and opened.bytes == expected_bytes,
        f"{label} exact bytes drifted from the retry amendment.",
    )
    _require(
        file_stat.st_uid == os.getuid()
        and file_stat.st_nlink == 1
        and stat.S_IMODE(file_stat.st_mode) == 0o600,
        f"{label} ownership, link count, or mode is unsafe.",
    )


def _opened_public_binding(
    opened: attestation.OpenedRegularFile,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Attested evidence envelope is missing.")
    envelope = cast(Mapping[str, Any], envelope)
    return {
        "path": str(opened.path),
        "sha256": opened.sha256,
        "bytes": opened.bytes,
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": envelope.get("mac"),
    }


def _assert_exact_quarantine_inventory() -> None:
    root = Path(os.path.abspath(SUPERSEDED_OUTPUT_ROOT))
    _require(root.is_dir() and not root.is_symlink(), "Quarantined calibration root is unsafe.")
    expected = {
        root / SUPERSEDED_MATRIX_SUMMARY_NAME,
        root / "s55",
        root / "s55" / "seed-6071406",
        root / "s55" / "seed-6071406" / CELL_CLAIM_NAME,
        root / "s55" / "seed-6071406" / "s55-calibration.json",
    }
    observed: set[Path] = set()
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"Quarantined calibration tree contains a symlink: {item}")
        observed.add(Path(os.path.abspath(item)))
    _require(
        observed == expected,
        "Quarantined calibration tree contains missing or extra evidence.",
    )


def _validate_quarantine_evidence(
    *,
    legacy_context: training_matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
    training_matrix_summary_path: Path,
    training_matrix_payload: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    ledger_records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> ValidatedQuarantineEvidence:
    _assert_exact_quarantine_inventory()
    root = Path(os.path.abspath(SUPERSEDED_OUTPUT_ROOT))
    matrix_path = root / SUPERSEDED_MATRIX_SUMMARY_NAME
    claim_path = root / "s55" / "seed-6071406" / CELL_CLAIM_NAME
    artifact_path = root / "s55" / "seed-6071406" / "s55-calibration.json"
    opened_matrix = attestation.open_regular_nofollow(matrix_path)
    opened_claim = attestation.open_regular_nofollow(claim_path)
    opened_artifact = attestation.open_regular_nofollow(artifact_path)
    opened_training_matrix = attestation.open_regular_nofollow(training_matrix_summary_path)
    opened_legacy_manifest = attestation.open_regular_nofollow(legacy_context.manifest_path)
    opened_checkpoint: attestation.OpenedRegularFile | None = None
    try:
        _require(
            opened_legacy_manifest.path == legacy_context.manifest_path
            and opened_legacy_manifest.sha256 == contract.V1_1_MANIFEST_SHA256
            and opened_legacy_manifest.sha256 == legacy_context.manifest_binding["sha256"],
            "Revision 1.1 manifest file binding drifted during quarantine validation.",
        )
        _assert_safe_quarantine_file(
            opened_matrix,
            expected_sha256=contract.V1_1_CALIBRATION_MATRIX_SHA256,
            expected_bytes=8290,
            label="Quarantined calibration matrix ledger",
        )
        _assert_safe_quarantine_file(
            opened_claim,
            expected_sha256=contract.V1_1_CALIBRATION_CLAIM_SHA256,
            expected_bytes=360,
            label="Quarantined calibration claim",
        )
        _assert_safe_quarantine_file(
            opened_artifact,
            expected_sha256=contract.V1_1_CALIBRATION_ARTIFACT_SHA256,
            expected_bytes=16_484_030,
            label="Quarantined calibration artifact",
        )
        _assert_safe_quarantine_file(
            opened_training_matrix,
            expected_sha256=contract.V1_1_TRAINING_MATRIX_SHA256,
            expected_bytes=18_032,
            label="Terminal revision 1.1 training matrix ledger",
        )
        try:
            matrix_payload = json.loads(opened_matrix.read_bytes().decode("utf-8"))
            claim_payload = json.loads(opened_claim.read_bytes().decode("utf-8"))
            artifact_payload = json.loads(opened_artifact.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Quarantined calibration evidence contains invalid JSON.") from error
        _require(
            isinstance(matrix_payload, dict)
            and isinstance(claim_payload, dict)
            and isinstance(artifact_payload, dict),
            "Quarantined calibration evidence must contain JSON objects.",
        )
        _require(
            set(matrix_payload) == SUPERSEDED_CALIBRATION_MATRIX_FIELDS,
            "Quarantined calibration matrix schema drifted.",
        )
        _validate_payload_digest(matrix_payload)
        matrix_envelope = matrix_payload.get("attestation")
        _require(isinstance(matrix_envelope, Mapping), "Quarantined matrix attestation is missing.")
        matrix_semantic = dict(matrix_payload)
        matrix_semantic.pop("attestation")
        attestation.verify_attestation(
            matrix_semantic,
            cast(Mapping[str, Any], matrix_envelope),
            trust_root=trust_root,
            purpose="p2-direct-soft-lag-calibration-matrix-v1",
        )
        _require(
            matrix_payload.get("schema_version") == 5
            and matrix_payload.get("experiment_id")
            == "p2-post-rank-direct-soft-lag-calibration-matrix-v1"
            and matrix_payload.get("status") == "in_progress"
            and matrix_payload.get("completed_cells") == 0
            and matrix_payload.get("expected_cells") == EXPECTED_CELLS
            and matrix_payload.get("cells") == []
            and matrix_payload.get("cell_decisions") == {}
            and matrix_payload.get("terminal_decision") is None
            and matrix_payload.get("source") == legacy_context.source
            and matrix_payload.get("manifest") == legacy_context.manifest_binding,
            "Quarantined calibration matrix is not the exact failed empty prefix.",
        )
        _require(
            matrix_payload.get("execution_environment")
            == execution_environment.validate_execution_environment(
                cast(Mapping[str, Any], matrix_payload["execution_environment"])
            ),
            "Quarantined calibration execution environment is invalid.",
        )
        legacy_gpu_lease = _validate_gpu_lease_binding(matrix_payload.get("gpu_lease"))
        _require(
            legacy_gpu_lease["path"] == str(_canonical_gpu_lock_path(RETRY_GPU_LOCK_PATH))
            and legacy_gpu_lease["selected_device_class"]
            == execution_environment.selected_device_class(
                cast(Mapping[str, Any], matrix_payload["execution_environment"])
            )
            and legacy_gpu_lease["selected_device_routing_identity"]
            == execution_environment.selected_device_routing_identity(
                cast(Mapping[str, Any], matrix_payload["execution_environment"])
            ),
            "Quarantined calibration GPU scheduler lease drifted from the registered retry.",
        )
        _require(
            set(claim_payload)
            == {
                "schema_version",
                "semantics",
                "coordinate",
                "launch_nonce",
                "pid",
                "created_time_ns",
            }
            and claim_payload.get("schema_version") == 1
            and claim_payload.get("semantics") == CELL_CLAIM_SEMANTICS
            and claim_payload.get("coordinate")
            == {
                "scale": "s55",
                "training_seed": 6071406,
                "calibration_seed": 7071406,
                "evaluation_seed_reserved": 10071406,
            }
            and claim_payload.get("launch_nonce") == contract.V1_1_CALIBRATION_LAUNCH_NONCE
            and type(claim_payload.get("pid")) is int
            and cast(int, claim_payload["pid"]) > 0
            and type(claim_payload.get("created_time_ns")) is int
            and cast(int, claim_payload["created_time_ns"]) > 0,
            "Quarantined calibration claim schema or coordinate drifted.",
        )
        scale, training_seed, calibration_seed, evaluation_seed = RETRY_COORDINATE
        summary_path, summary, checkpoint = _validate_training_input(
            training_output_root=TRAINING_OUTPUT_ROOT,
            scale=scale,
            training_seed=training_seed,
            context=legacy_context,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
            ledger_record=ledger_records[(scale, training_seed)],
        )
        checkpoint_path = Path(cast(str, checkpoint["path"]))
        opened_checkpoint = attestation.open_regular_nofollow(checkpoint_path)
        _assert_safe_quarantine_file(
            opened_checkpoint,
            expected_sha256=contract.SUPERSEDED_CHECKPOINT_SHA256,
            expected_bytes=626_727_758,
            label="Retry-coordinate checkpoint",
        )
        calibration.validate_calibration_artifact(
            artifact_payload,
            verify_bindings=True,
            trust_root=trust_root,
            expected_manifest_experiment_id="p2-post-rank-direct-controller-v1.1",
        )
        _require(
            artifact_payload.get("scale") == scale
            and artifact_payload.get("training_seed") == training_seed
            and artifact_payload.get("calibration_seed") == calibration_seed
            and artifact_payload.get("evaluation_seed_reserved") == evaluation_seed
            and artifact_payload.get("source") == legacy_context.source
            and artifact_payload.get("manifest") == legacy_context.manifest_binding
            and artifact_payload.get("checkpoint") == _expected_checkpoint_binding(checkpoint)
            and artifact_payload.get("training_summary")
            == _expected_training_binding(
                summary_path=summary_path,
                summary=summary,
                checkpoint=checkpoint,
                scale=scale,
                training_seed=training_seed,
                training_matrix_summary_path=training_matrix_summary_path,
                training_matrix_payload=training_matrix_payload,
                ledger_record=ledger_records[(scale, training_seed)],
                result_context=legacy_context,
                training_context=legacy_context,
            )
            and artifact_payload.get("environment") == matrix_payload.get("execution_environment")
            and artifact_payload.get("terminal_decision") == "GO"
            and artifact_payload.get("payload_sha256")
            == contract.V1_1_CALIBRATION_ARTIFACT_PAYLOAD_SHA256
            and cast(Mapping[str, Any], artifact_payload.get("attestation", {})).get("mac")
            == contract.V1_1_CALIBRATION_ARTIFACT_ATTESTATION_MAC,
            "Quarantined calibration artifact binding drifted beyond the registered path bug.",
        )
        opened_matrix.assert_unchanged()
        opened_claim.assert_unchanged()
        opened_artifact.assert_unchanged()
        opened_training_matrix.assert_unchanged()
        opened_legacy_manifest.assert_unchanged()
        opened_checkpoint.assert_unchanged()
        matrix_binding = _opened_public_binding(opened_matrix, matrix_payload)
        claim_binding = {
            "path": str(opened_claim.path),
            "sha256": opened_claim.sha256,
            "bytes": opened_claim.bytes,
            "launch_nonce": claim_payload["launch_nonce"],
            "coordinate": claim_payload["coordinate"],
        }
        artifact_binding = _opened_public_binding(opened_artifact, artifact_payload)
        artifact_binding["terminal_decision"] = artifact_payload["terminal_decision"]
        training_binding = _opened_public_binding(
            opened_training_matrix,
            training_matrix_payload,
        )
        checkpoint_binding = {
            "path": str(opened_checkpoint.path),
            "sha256": opened_checkpoint.sha256,
            "bytes": opened_checkpoint.bytes,
            "authenticated_path_spelling": checkpoint["path"],
        }
        return ValidatedQuarantineEvidence(
            legacy_context=legacy_context,
            legacy_manifest_file_binding={
                "path": str(opened_legacy_manifest.path),
                "sha256": opened_legacy_manifest.sha256,
                "bytes": opened_legacy_manifest.bytes,
            },
            matrix_ledger=matrix_payload,
            matrix_ledger_binding=matrix_binding,
            claim=claim_payload,
            claim_binding=claim_binding,
            artifact=artifact_payload,
            artifact_binding=artifact_binding,
            training_matrix_binding=training_binding,
            checkpoint_binding=checkpoint_binding,
            execution_environment=execution_environment.validate_execution_environment(
                cast(Mapping[str, Any], matrix_payload["execution_environment"])
            ),
        )
    finally:
        if opened_checkpoint is not None:
            opened_checkpoint.close()
        opened_legacy_manifest.close()
        opened_training_matrix.close()
        opened_artifact.close()
        opened_claim.close()
        opened_matrix.close()


def _retry_admission_semantic_payload(
    *,
    output_root: Path,
    context: training_matrix.FrozenContext,
    evidence: ValidatedQuarantineEvidence,
    incident_report_binding: Mapping[str, Any],
) -> dict[str, Any]:
    scale, training_seed, calibration_seed, evaluation_seed = RETRY_COORDINATE
    opened_current_manifest = attestation.open_regular_nofollow(context.manifest_path)
    try:
        _require(
            opened_current_manifest.sha256 == context.manifest_binding["sha256"],
            "Current manifest bytes drifted while building the retry admission.",
        )
        current_manifest_binding = {
            **context.manifest_binding,
            "bytes": opened_current_manifest.bytes,
        }
        opened_current_manifest.assert_unchanged()
    finally:
        opened_current_manifest.close()
    return {
        "schema_version": 1,
        "admission_id": RETRY_ADMISSION_ID,
        "status": "terminal",
        "reason": RETRY_ADMISSION_REASON,
        "coordinate": {
            "scale": scale,
            "training_seed": training_seed,
            "calibration_seed": calibration_seed,
            "evaluation_seed_reserved": evaluation_seed,
        },
        "observed_terminal_decision": "GO",
        "superseded_manifest": {
            **evidence.legacy_context.manifest_binding,
            "bytes": evidence.legacy_manifest_file_binding["bytes"],
        },
        "superseded_matrix_ledger": evidence.matrix_ledger_binding,
        "preserved_claim": evidence.claim_binding,
        "preserved_calibration_artifact": evidence.artifact_binding,
        "terminal_training_matrix_ledger": evidence.training_matrix_binding,
        "checkpoint": evidence.checkpoint_binding,
        "execution_environment": evidence.execution_environment,
        "current_manifest": current_manifest_binding,
        "current_source": context.source,
        "incident_report": dict(incident_report_binding),
        "quarantine_rule": {
            "root": str(Path(os.path.abspath(SUPERSEDED_OUTPUT_ROOT))),
            "immutable_forever": True,
            "legacy_artifact_admissible_as_v1_2_result": False,
        },
        "retry_rule": {
            "output_root": str(Path(os.path.abspath(output_root))),
            "scheduler_gpu_lock_path": str(_canonical_gpu_lock_path(RETRY_GPU_LOCK_PATH)),
            "failed_attempt_gpu_lease": dict(evidence.matrix_ledger["gpu_lease"]),
            "retry_limit": 1,
            "coordinate_count": 1,
            "authorization_registered_after_result_disclosure": True,
            "authorization_basis": "parent-only-checkpoint-path-representation-false-negative",
            "counterfactual_outcome_independence_claimed": False,
            "old_result_admitted_to_amended_cohort": False,
            "retry_consumed_at_admission_commit": True,
            "first_launch_requires_same_process_that_created_admission": True,
            "read_only_preflight_completed_before_admission_commit": True,
            "restart_before_first_ledger_promotion": "terminal-fail-closed-no-retry",
            "owner_controlled_deletion_or_filesystem_rollback": (
                "outside-threat-model-and-invalidates-evidence"
            ),
            "general_retry_policy_created": False,
        },
        "scientific_subprocesses_started_at_creation": 0,
    }


def _admission_encoded_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_mkdir(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    if absolute.exists():
        _require(
            absolute.is_dir() and not absolute.is_symlink(),
            "Durable calibration directory is unsafe.",
        )
        return
    parent = absolute.parent
    _require(
        parent.is_dir() and not parent.is_symlink(),
        "Durable calibration directory parent is missing or unsafe.",
    )
    absolute.mkdir(mode=0o700)
    _fsync_directory(absolute)
    _fsync_directory(parent)


def _exclusive_write_retry_admission(path: Path, payload: Mapping[str, Any]) -> None:
    _durable_mkdir(path.parent)
    staging = path.parent / RETRY_ADMISSION_STAGING_NAME
    encoded = _admission_encoded_bytes(payload)
    if os.path.lexists(staging):
        staged = attestation.open_regular_nofollow(staging)
        try:
            _require(
                staged.read_bytes() == encoded,
                "Retry-admission staging bytes differ from the registered payload.",
            )
        finally:
            staged.close()
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        no_follow = getattr(os, "O_NOFOLLOW", None)
        _require(no_follow is not None, "Retry admission requires O_NOFOLLOW support.")
        descriptor = os.open(staging, flags | cast(int, no_follow), 0o600)
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                _require(written > 0, "Retry-admission staging write made no progress.")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
    if not os.path.lexists(path):
        try:
            os.link(staging, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError("Retry admission already exists and cannot be recreated.") from error
        _fsync_directory(path.parent)
    final = attestation.open_regular_nofollow(path)
    staged = attestation.open_regular_nofollow(staging)
    try:
        final_stat = os.fstat(final.file_descriptor)
        staged_stat = os.fstat(staged.file_descriptor)
        _require(
            final.read_bytes() == encoded
            and staged.read_bytes() == encoded
            and (final_stat.st_dev, final_stat.st_ino)
            == (staged_stat.st_dev, staged_stat.st_ino)
            and final_stat.st_uid == os.getuid()
            and final_stat.st_nlink == 2
            and stat.S_IMODE(final_stat.st_mode) == 0o600,
            "Committed retry admission staging/final inode is unsafe.",
        )
        os.fsync(final.file_descriptor)
    finally:
        staged.close()
        final.close()
    staging.unlink()
    _fsync_directory(path.parent)
    final = attestation.open_regular_nofollow(path)
    try:
        final_stat = os.fstat(final.file_descriptor)
        _require(
            final.read_bytes() == encoded
            and final_stat.st_uid == os.getuid()
            and final_stat.st_nlink == 1
            and stat.S_IMODE(final_stat.st_mode) == 0o600,
            "Final retry admission ownership, link count, mode, or bytes drifted.",
        )
        final.assert_unchanged()
    finally:
        final.close()


def _retry_admission_public_binding(
    opened: attestation.OpenedRegularFile,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    envelope = cast(Mapping[str, Any], payload["attestation"])
    binding = {
        "path": str(opened.path),
        "sha256": opened.sha256,
        "bytes": opened.bytes,
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": envelope.get("mac"),
        "admission_id": payload.get("admission_id"),
        "coordinate": payload.get("coordinate"),
        "preserved_claim_sha256": cast(Mapping[str, Any], payload["preserved_claim"]).get(
            "sha256"
        ),
        "preserved_artifact_sha256": cast(
            Mapping[str, Any], payload["preserved_calibration_artifact"]
        ).get("sha256"),
    }
    _require(
        set(binding) == RETRY_ADMISSION_BINDING_FIELDS,
        "Retry-admission public binding schema drifted.",
    )
    return binding


def _assert_retry_admission_file_binding(binding: Mapping[str, Any]) -> None:
    _require(
        set(binding) == RETRY_ADMISSION_BINDING_FIELDS,
        "Retry-admission file binding schema drifted.",
    )
    path = binding.get("path")
    _require(isinstance(path, str) and bool(path), "Retry-admission path binding is invalid.")
    opened = attestation.open_regular_nofollow(Path(cast(str, path)))
    try:
        metadata = os.fstat(opened.file_descriptor)
        _require(
            opened.sha256 == binding.get("sha256")
            and opened.bytes == binding.get("bytes")
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600,
            "Retry-admission file changed, disappeared, or became unsafe.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()


def _load_retry_admission(
    *,
    path: Path,
    output_root: Path,
    context: training_matrix.FrozenContext,
    evidence: ValidatedQuarantineEvidence,
    incident_report_binding: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
) -> ValidatedRetryAdmission:
    opened = attestation.open_regular_nofollow(path)
    try:
        file_stat = os.fstat(opened.file_descriptor)
        _require(
            file_stat.st_uid == os.getuid()
            and file_stat.st_nlink == 1
            and stat.S_IMODE(file_stat.st_mode) == 0o600,
            "Retry admission ownership, link count, or mode is unsafe.",
        )
        raw = opened.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Retry admission is invalid JSON.") from error
        _require(isinstance(payload, dict), "Retry admission must be a JSON object.")
        _require(set(payload) == RETRY_ADMISSION_FIELDS, "Retry admission schema drifted.")
        _validate_payload_digest(payload)
        semantic = dict(payload)
        envelope = semantic.pop("attestation")
        _require(isinstance(envelope, Mapping), "Retry admission attestation is missing.")
        attestation.verify_attestation(
            semantic,
            cast(Mapping[str, Any], envelope),
            trust_root=trust_root,
            purpose=RETRY_ADMISSION_ATTESTATION_PURPOSE,
        )
        expected = _attested_payload(
            _retry_admission_semantic_payload(
                output_root=output_root,
                context=context,
                evidence=evidence,
                incident_report_binding=incident_report_binding,
            ),
            trust_root=trust_root,
            purpose=RETRY_ADMISSION_ATTESTATION_PURPOSE,
        )
        _require(payload == expected, "Retry admission semantic evidence binding drifted.")
        _require(
            raw == _admission_encoded_bytes(payload),
            "Retry admission bytes are not in the one canonical encoding.",
        )
        opened.assert_unchanged()
        public_binding = _retry_admission_public_binding(opened, payload)
        return ValidatedRetryAdmission(
            payload=payload,
            public_binding=public_binding,
            evidence=evidence,
        )
    finally:
        opened.close()


def _load_or_create_retry_admission(
    *,
    output_root: Path,
    matrix_summary: Path,
    context: training_matrix.FrozenContext,
    legacy_context: training_matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
    training_matrix_summary_path: Path,
    training_matrix_payload: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    ledger_records: Mapping[tuple[str, int], Mapping[str, Any]],
    matrix_lock: _MatrixLockLease,
    gpu_lease: gpu_lock.GPULockLease,
    device_guard: gpu_lock.GPULockLease,
    expected_execution_environment: Mapping[str, Any],
) -> tuple[ValidatedRetryAdmission | None, bool]:
    admission_path = Path(os.path.abspath(output_root)) / RETRY_ADMISSION_NAME
    if not REQUIRE_RETRY_ADMISSION:
        _require(
            not os.path.lexists(admission_path),
            "Test-only retry-admission bypass may not coexist with an admission.",
        )
        return None, False
    matrix_lock.assert_held()
    gpu_lease.assert_held()
    device_guard.assert_held()
    evidence = _validate_quarantine_evidence(
        legacy_context=legacy_context,
        trust_root=trust_root,
        training_matrix_summary_path=training_matrix_summary_path,
        training_matrix_payload=training_matrix_payload,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
    )
    _require(
        dict(expected_execution_environment) == evidence.execution_environment,
        "Revision 1.2 retry requires the exact failed-attempt execution environment before "
        "its one-shot admission may be committed.",
    )
    report_path = REPOSITORY_ROOT / contract.CALIBRATION_PATH_AMENDMENT_REPORT_PATH
    opened_report = attestation.open_regular_nofollow(report_path)
    try:
        _require(
            opened_report.sha256 == contract.CALIBRATION_PATH_AMENDMENT_REPORT_SHA256,
            "Calibration path-amendment report bytes drifted.",
        )
        incident_report_binding = {
            "path": str(opened_report.path),
            "sha256": opened_report.sha256,
            "bytes": opened_report.bytes,
        }
        opened_report.assert_unchanged()
    finally:
        opened_report.close()
    if os.path.lexists(admission_path):
        if os.path.lexists(admission_path.parent / RETRY_ADMISSION_STAGING_NAME):
            expected_payload = _attested_payload(
                _retry_admission_semantic_payload(
                    output_root=output_root,
                    context=context,
                    evidence=evidence,
                    incident_report_binding=incident_report_binding,
                ),
                trust_root=trust_root,
                purpose=RETRY_ADMISSION_ATTESTATION_PURPOSE,
            )
            _exclusive_write_retry_admission(admission_path, expected_payload)
        return (
            _load_retry_admission(
                path=admission_path,
                output_root=output_root,
                context=context,
                evidence=evidence,
                incident_report_binding=incident_report_binding,
                trust_root=trust_root,
            ),
            False,
        )
    _require(
        not os.path.lexists(matrix_summary),
        "Amended calibration matrix exists without its retry admission.",
    )
    if output_root.exists():
        allowed_staging = output_root / RETRY_ADMISSION_STAGING_NAME
        _require(
            all(item == allowed_staging for item in output_root.iterdir()),
            "Retry admission must precede every amended calibration artifact.",
        )
    payload = _attested_payload(
        _retry_admission_semantic_payload(
            output_root=output_root,
            context=context,
            evidence=evidence,
            incident_report_binding=incident_report_binding,
        ),
        trust_root=trust_root,
        purpose=RETRY_ADMISSION_ATTESTATION_PURPOSE,
    )
    matrix_lock.assert_held()
    gpu_lease.assert_held()
    device_guard.assert_held()
    _exclusive_write_retry_admission(admission_path, payload)
    matrix_lock.assert_held()
    gpu_lease.assert_held()
    device_guard.assert_held()
    evidence = _validate_quarantine_evidence(
        legacy_context=legacy_context,
        trust_root=trust_root,
        training_matrix_summary_path=training_matrix_summary_path,
        training_matrix_payload=training_matrix_payload,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
    )
    return (
        _load_retry_admission(
            path=admission_path,
            output_root=output_root,
            context=context,
            evidence=evidence,
            incident_report_binding=incident_report_binding,
            trust_root=trust_root,
        ),
        True,
    )


@contextmanager
def _held_retry_evidence_snapshots(
    admission: ValidatedRetryAdmission | None,
) -> Iterator[None]:
    if admission is None:
        yield
        return
    _assert_exact_quarantine_inventory()
    bound_items = [
        admission.public_binding,
        admission.payload["superseded_manifest"],
        admission.payload["current_manifest"],
        admission.payload["superseded_matrix_ledger"],
        admission.payload["preserved_claim"],
        admission.payload["preserved_calibration_artifact"],
        admission.payload["terminal_training_matrix_ledger"],
        admission.payload["checkpoint"],
        admission.payload["incident_report"],
    ]
    opened_files: list[attestation.OpenedRegularFile] = []
    try:
        for binding in bound_items:
            _require(isinstance(binding, Mapping), "Retry snapshot binding is invalid.")
            path = binding.get("path")
            _require(isinstance(path, str), "Retry snapshot path binding is invalid.")
            opened = attestation.open_regular_nofollow(Path(path))
            _require(
                opened.sha256 == binding.get("sha256")
                and opened.bytes == binding.get("bytes"),
                "Retry evidence changed before child launch.",
            )
            opened_files.append(opened)
        for opened in opened_files:
            opened.assert_unchanged()
        _assert_exact_quarantine_inventory()
        yield
        _assert_exact_quarantine_inventory()
        for opened in opened_files:
            opened.assert_unchanged()
    finally:
        for opened in reversed(opened_files):
            opened.close()


@contextmanager
def _held_matrix_ledger_snapshot(
    path: Path,
    *,
    expected_payload: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
) -> Iterator[None]:
    """Hold the exact validated prefix ledger inode across one scientific child."""

    opened = attestation.open_regular_nofollow(path)
    try:
        metadata = os.fstat(opened.file_descriptor)
        _require(
            metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o600,
            "Calibration matrix prefix ledger ownership, link count, or mode is unsafe.",
        )
        raw = opened.read_bytes()
        expected_raw = (
            json.dumps(expected_payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        _require(raw == expected_raw, "Calibration matrix prefix bytes differ from the expected prefix.")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Calibration matrix prefix ledger is invalid JSON.") from error
        _require(
            isinstance(payload, dict) and payload == dict(expected_payload),
            "Calibration matrix prefix ledger differs from the validated in-memory prefix.",
        )
        _validate_payload_digest(payload)
        envelope = payload.get("attestation")
        _require(isinstance(envelope, Mapping), "Calibration matrix prefix attestation is missing.")
        semantic = dict(payload)
        semantic.pop("attestation")
        attestation.verify_attestation(
            semantic,
            cast(Mapping[str, Any], envelope),
            trust_root=trust_root,
            purpose=MATRIX_ATTESTATION_PURPOSE,
        )
        opened.assert_unchanged()
        yield
        opened.assert_unchanged()
    finally:
        opened.close()


def load_retry_admission_for_downstream(
    *,
    output_root: Path,
    context: training_matrix.FrozenContext,
    training_context: training_matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
    training_matrix_summary_path: Path,
    training_matrix_payload: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    ledger_records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> ValidatedRetryAdmission | None:
    if not REQUIRE_RETRY_ADMISSION:
        return None
    evidence = _validate_quarantine_evidence(
        legacy_context=training_context,
        trust_root=trust_root,
        training_matrix_summary_path=training_matrix_summary_path,
        training_matrix_payload=training_matrix_payload,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
    )
    report_path = REPOSITORY_ROOT / contract.CALIBRATION_PATH_AMENDMENT_REPORT_PATH
    opened_report = attestation.open_regular_nofollow(report_path)
    try:
        _require(
            opened_report.sha256 == contract.CALIBRATION_PATH_AMENDMENT_REPORT_SHA256,
            "Calibration path-amendment report bytes drifted.",
        )
        report_binding = {
            "path": str(opened_report.path),
            "sha256": opened_report.sha256,
            "bytes": opened_report.bytes,
        }
        opened_report.assert_unchanged()
    finally:
        opened_report.close()
    return _load_retry_admission(
        path=Path(os.path.abspath(output_root)) / RETRY_ADMISSION_NAME,
        output_root=output_root,
        context=context,
        evidence=evidence,
        incident_report_binding=report_binding,
        trust_root=trust_root,
    )


def _load_terminal_training_ledger(
    *,
    training_output_root: Path,
    context: training_matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    """Authenticate the terminal 10-cell training ledger before any calibration launch."""

    canonical_trainer = training_matrix._canonical_train_script(training_matrix.TRAIN_SCRIPT)
    opened_trainer, trainer_snapshot = training_matrix._open_canonical_trainer(canonical_trainer)
    ledger_path = training_output_root / training_matrix.MATRIX_SUMMARY.name
    _require(ledger_path.is_file(), "Terminal direct-training matrix ledger is missing.")
    _require(not ledger_path.is_symlink(), "Training matrix ledger may not be a symbolic link.")
    opened_ledger = attestation.open_regular_nofollow(ledger_path)
    try:
        try:
            payload = json.loads(opened_ledger.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Terminal direct-training matrix ledger is invalid JSON.") from error
        _require(isinstance(payload, dict), "Training matrix ledger must be a JSON object.")
        trainer_binding = trainer_snapshot.public_binding
        validated = training_matrix.validate_matrix_summary(
            payload,
            output_root=training_output_root,
            train_script=canonical_trainer,
            context=context,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
        )
        _require(
            payload.get("status") == "terminal"
            and payload.get("completed_runs") == payload.get("expected_runs") == EXPECTED_CELLS
            and len(validated) == EXPECTED_CELLS,
            "Calibration requires a terminal attested 10-cell training matrix ledger.",
        )
        coordinates = [(scale, training_seed) for scale, training_seed, _, _ in _coordinates()]
        ledger_records: dict[tuple[str, int], dict[str, Any]] = {}
        for record, coordinate in zip(validated, coordinates, strict=True):
            _require(
                (record.get("scale"), record.get("seed")) == coordinate,
                "Terminal training ledger coordinate order drifted.",
            )
            ledger_records[coordinate] = dict(record)
        opened_ledger.assert_unchanged()
        opened_trainer.assert_unchanged()
        training_matrix.assert_frozen_context_reference_unchanged(
            context,
            trust_root=trust_root,
        )
        return ledger_path, payload, trainer_binding, ledger_records
    finally:
        opened_ledger.close()
        opened_trainer.close()


def _preflight_output_tree(
    *,
    output_root: Path,
    matrix_summary: Path,
    completed_cells: int,
    retry_admission_path: Path | None = None,
) -> None:
    """Reject every non-prefix or non-inventory output before launch/publication."""

    root = Path(os.path.abspath(output_root))
    summary = Path(os.path.abspath(matrix_summary))
    _require(not output_root.is_symlink(), "Calibration output root may not be a symbolic link.")
    _require(not matrix_summary.is_symlink(), "Calibration matrix may not be a symbolic link.")
    _require(
        summary.is_relative_to(root),
        "Calibration matrix summary must be stored under its output root.",
    )
    if not root.exists():
        _require(completed_cells == 0, "Completed calibration prefix lost its output root.")
        return
    _require(root.is_dir(), "Calibration output root is not a directory.")

    allowed: set[Path] = {summary}
    if retry_admission_path is not None:
        admission = Path(os.path.abspath(retry_admission_path))
        _require(
            admission.parent == root,
            "Calibration retry admission must be stored directly under the output root.",
        )
        allowed.add(admission)
    cursor = summary.parent
    while cursor != root:
        _require(cursor.is_relative_to(root), "Calibration summary parent escaped output root.")
        allowed.add(cursor)
        cursor = cursor.parent

    coordinates = _coordinates()
    for index, (scale, training_seed, _, _) in enumerate(coordinates):
        scale_dir = root / scale
        output_dir = scale_dir / f"seed-{training_seed}"
        artifact_path = output_dir / f"{scale}-calibration.json"
        claim_path = output_dir / CELL_CLAIM_NAME
        allowed.update((scale_dir, output_dir))
        if index < completed_cells:
            allowed.add(artifact_path)
            _require(
                output_dir.is_dir() and not output_dir.is_symlink(),
                f"Completed calibration directory is missing or unsafe: {scale}/{training_seed}.",
            )
            _require(
                artifact_path.is_file() and not artifact_path.is_symlink(),
                f"Previously completed calibration artifact disappeared: {scale}/{training_seed}.",
            )
            _require(
                not os.path.lexists(claim_path),
                f"Completed calibration cell retained a claim: {scale}/{training_seed}.",
            )
        elif output_dir.exists():
            _require(
                output_dir.is_dir() and not output_dir.is_symlink(),
                f"Calibration output path is not a safe directory: {scale}/{training_seed}.",
            )
            _require(
                not any(output_dir.iterdir()),
                f"Refusing orphaned or stale calibration output: {output_dir}",
            )

    for item in root.rglob("*"):
        absolute = Path(os.path.abspath(item))
        _require(not item.is_symlink(), f"Calibration output tree contains a symlink: {item}")
        _require(
            absolute in allowed,
            f"Calibration output tree contains an unregistered orphan: {item}",
        )


def _preflight_training_inputs(
    *,
    training_output_root: Path,
    context: training_matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
    trainer_binding: Mapping[str, Any],
    ledger_records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[tuple[str, int], tuple[Path, dict[str, Any], dict[str, Any]]]:
    validated: dict[tuple[str, int], tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for scale, training_seed, _, _ in _coordinates():
        validated[(scale, training_seed)] = _validate_training_input(
            training_output_root=training_output_root,
            scale=scale,
            training_seed=training_seed,
            context=context,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
            ledger_record=ledger_records[(scale, training_seed)],
        )
    return validated


def _run_matrix_locked(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = TRAINING_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path = MATRIX_SUMMARY,
    calibration_script: Path = CALIBRATION_SCRIPT,
    matrix_lock_binding: Mapping[str, Any],
    matrix_lock: _MatrixLockLease,
    gpu_lease: gpu_lock.GPULockLease,
    device_guard: gpu_lock.GPULockLease,
    gpu_lease_binding: Mapping[str, Any],
    frozen_execution_environment: Mapping[str, Any],
    attestation_key_path: Path | None,
) -> dict[str, Any]:
    matrix_lock.assert_held()
    gpu_lease.assert_held()
    device_guard.assert_held()
    gpu_lease.assert_held()
    device_guard.assert_held()
    lock_path = str(matrix_lock_binding.get("path"))
    _require(
        str(matrix_lock.path) == lock_path,
        "Calibration matrix lock lease does not match its public binding.",
    )
    with _ACTIVE_MATRIX_LOCKS_GUARD:
        lock_owned = lock_path in _ACTIVE_MATRIX_LOCKS
    _require(lock_owned, "Calibration matrix execution requires the exclusive process lock.")
    if REQUIRE_RETRY_ADMISSION:
        expected_retry_lock = str(_canonical_gpu_lock_path(RETRY_GPU_LOCK_PATH))
        _require(
            str(_canonical_gpu_lock_path(gpu_lease.path)) == expected_retry_lock
            and gpu_lease_binding.get("path") == expected_retry_lock,
            "Revision 1.2 retry is not holding the frozen failed-attempt GPU scheduler lock.",
        )
    canonical_script = _canonical_calibration_script(calibration_script)
    context = training_matrix.establish_frozen_context(manifest_path)
    if REQUIRE_RETRY_ADMISSION:
        expected_manifest_path = (
            contract.MANIFEST_PATH
            if contract.MANIFEST_PATH.is_absolute()
            else REPOSITORY_ROOT / contract.MANIFEST_PATH
        ).resolve()
        _require(
            context.manifest_path == expected_manifest_path
            and context.manifest_binding.get("experiment_id") == contract.EXPERIMENT_ID,
            "Revision 1.2 retry requires the canonical current manifest context.",
        )
    _assert_calibration_implementation_binding(context)
    raw_manifest_attestation = context.manifest_binding.get("attestation")
    _require(
        isinstance(raw_manifest_attestation, Mapping),
        "Manifest attestation binding is missing.",
    )
    manifest_attestation = cast(Mapping[str, Any], raw_manifest_attestation)
    expected_key_id = manifest_attestation.get("key_id")
    _require(contract.is_sha256(expected_key_id), "Manifest attestation key ID is invalid.")
    if attestation_key_path is None:
        trust_root = attestation.trust_root_from_environment(
            repository_root=REPOSITORY_ROOT,
            artifact_roots=(training_output_root, output_root),
            expected_key_id=cast(str, expected_key_id),
        )
    else:
        trust_root = attestation.load_trust_root(
            attestation_key_path,
            repository_root=REPOSITORY_ROOT,
            artifact_roots=(training_output_root, output_root),
            expected_key_id=cast(str, expected_key_id),
        )
    script_fd, script_snapshot = _open_canonical_script(canonical_script)
    os.close(script_fd)
    script_binding = script_snapshot.public_binding
    training_context = (
        _load_v1_1_context(trust_root=trust_root) if REQUIRE_RETRY_ADMISSION else context
    )
    (
        training_matrix_summary_path,
        training_matrix_payload,
        trainer_binding,
        ledger_records,
    ) = _load_terminal_training_ledger(
        training_output_root=training_output_root,
        context=training_context,
        trust_root=trust_root,
    )
    admission, retry_launch_authorized_in_this_process = _load_or_create_retry_admission(
        output_root=output_root,
        matrix_summary=matrix_summary,
        context=context,
        legacy_context=training_context,
        trust_root=trust_root,
        training_matrix_summary_path=training_matrix_summary_path,
        training_matrix_payload=training_matrix_payload,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
        matrix_lock=matrix_lock,
        gpu_lease=gpu_lease,
        device_guard=device_guard,
        expected_execution_environment=frozen_execution_environment,
    )
    retry_admission_binding = None if admission is None else admission.public_binding
    if admission is not None:
        _require(
            execution_environment.validate_execution_environment(frozen_execution_environment)
            == admission.evidence.execution_environment,
            "Revision 1.2 retry requires the exact failed-attempt execution environment.",
        )

    completed: list[dict[str, Any]] = []
    current_prefix_payload: dict[str, Any] | None = None
    if matrix_summary.exists():
        gpu_lease.assert_held()
        current_prefix_payload = _load_json(matrix_summary, label="calibration matrix")
        completed = validate_matrix_summary(
            current_prefix_payload,
            output_root=output_root,
            training_output_root=training_output_root,
            calibration_script=canonical_script,
            calibration_script_binding=script_binding,
            matrix_lock_binding=matrix_lock_binding,
            expected_gpu_lease=gpu_lease_binding,
            expected_execution_environment=frozen_execution_environment,
            context=context,
            training_context=training_context,
            trust_root=trust_root,
            training_matrix_summary_path=training_matrix_summary_path,
            training_matrix_payload=training_matrix_payload,
            trainer_binding=trainer_binding,
            ledger_records=ledger_records,
            retry_admission=retry_admission_binding,
        )
        matrix_lock.assert_held()

    if REQUIRE_RETRY_ADMISSION and len(completed) == 0:
        _require(
            retry_launch_authorized_in_this_process,
            "The sole calibration retry was consumed when its admission was committed by a "
            "prior process; an empty or rolled-back prefix may not launch it again.",
        )

    _preflight_output_tree(
        output_root=output_root,
        matrix_summary=matrix_summary,
        completed_cells=len(completed),
        retry_admission_path=(
            None if admission is None else Path(cast(str, admission.public_binding["path"]))
        ),
    )
    training_inputs = _preflight_training_inputs(
        training_output_root=training_output_root,
        context=training_context,
        trust_root=trust_root,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
    )
    training_matrix.assert_environment_unchanged(context)
    verification_fd, _ = _open_canonical_script(canonical_script, expected=script_snapshot)
    os.close(verification_fd)
    matrix_lock.assert_held()

    if not matrix_summary.exists():
        gpu_lease.assert_held()
        empty_prefix = _matrix_payload(
            completed,
            context=context,
            calibration_script_binding=script_binding,
            matrix_lock_binding=matrix_lock_binding,
            gpu_lease_binding=gpu_lease_binding,
            execution_environment_binding=frozen_execution_environment,
            trust_root=trust_root,
            retry_admission=retry_admission_binding,
        )
        _publish_matrix_ledger(
            matrix_summary,
            empty_prefix,
            matrix_lock=matrix_lock,
            trust_root=trust_root,
        )
        validated_empty_prefix = validate_matrix_summary(
            _load_json(matrix_summary, label="empty-prefix calibration matrix"),
            output_root=output_root,
            training_output_root=training_output_root,
            calibration_script=canonical_script,
            calibration_script_binding=script_binding,
            matrix_lock_binding=matrix_lock_binding,
            expected_gpu_lease=gpu_lease_binding,
            expected_execution_environment=frozen_execution_environment,
            context=context,
            training_context=training_context,
            trust_root=trust_root,
            training_matrix_summary_path=training_matrix_summary_path,
            training_matrix_payload=training_matrix_payload,
            trainer_binding=trainer_binding,
            ledger_records=ledger_records,
            retry_admission=retry_admission_binding,
        )
        _require(
            validated_empty_prefix == [],
            "Fresh amended calibration ledger is not the independently validated empty prefix.",
        )
        current_prefix_payload = empty_prefix
        gpu_lease.assert_held()

    _require(
        current_prefix_payload is not None,
        "Calibration matrix prefix was not established before child iteration.",
    )
    for index, (scale, training_seed, calibration_seed, evaluation_seed) in enumerate(
        _coordinates()
    ):
        matrix_lock.assert_held()
        gpu_lease.assert_held()
        training_matrix.assert_environment_unchanged(context)
        execution_environment.assert_exact_execution_environment(frozen_execution_environment)
        gpu_lease.assert_held()
        summary_path, summary, checkpoint = training_inputs[(scale, training_seed)]
        artifact_path = _artifact_path(output_root, scale, training_seed)
        checkpoint_path = Path(cast(str, checkpoint["path"]))
        command = build_calibration_command(
            calibration_script=canonical_script,
            artifact_path=artifact_path,
            manifest_path=context.manifest_path,
            training_manifest_path=training_context.manifest_path,
            training_summary_path=summary_path,
            training_matrix_summary_path=training_matrix_summary_path,
            checkpoint_path=checkpoint_path,
            scale=scale,
            training_seed=training_seed,
            device_index=cast(int, frozen_execution_environment["current_device_index"]),
            device_routing_identity=(
                execution_environment.selected_device_routing_identity(frozen_execution_environment)
            ),
        )

        new_record: dict[str, Any] | None = None
        if artifact_path.exists():
            _require(
                index < len(completed),
                f"Refusing orphaned pre-existing calibration output: {scale}/{training_seed}.",
            )
            artifact = load_and_validate_calibration_artifact(
                artifact_path,
                scale=scale,
                training_seed=training_seed,
                calibration_seed=calibration_seed,
                evaluation_seed=evaluation_seed,
                context=context,
                training_context=training_context,
                summary_path=summary_path,
                summary=summary,
                checkpoint=checkpoint,
                command=command,
                trust_root=trust_root,
                training_matrix_summary_path=training_matrix_summary_path,
                training_matrix_payload=training_matrix_payload,
                ledger_record=ledger_records[(scale, training_seed)],
            )
        else:
            _require(
                index >= len(completed),
                f"Previously completed calibration artifact disappeared: {scale}/{training_seed}.",
            )
            _assert_empty_cell_output(_cell_output_dir(output_root, scale, training_seed))
            summary_path, summary, checkpoint = _validate_training_input(
                training_output_root=training_output_root,
                scale=scale,
                training_seed=training_seed,
                context=training_context,
                trust_root=trust_root,
                trainer_binding=trainer_binding,
                ledger_record=ledger_records[(scale, training_seed)],
            )
            checkpoint_path = Path(cast(str, checkpoint["path"]))
            command = build_calibration_command(
                calibration_script=canonical_script,
                artifact_path=artifact_path,
                manifest_path=context.manifest_path,
                training_manifest_path=training_context.manifest_path,
                training_summary_path=summary_path,
                training_matrix_summary_path=training_matrix_summary_path,
                checkpoint_path=checkpoint_path,
                scale=scale,
                training_seed=training_seed,
                device_index=cast(int, frozen_execution_environment["current_device_index"]),
                device_routing_identity=(
                    execution_environment.selected_device_routing_identity(
                        frozen_execution_environment
                    )
                ),
            )
            launch_nonce = secrets.token_hex(32)
            with _exclusive_cell_claim(
                _cell_output_dir(output_root, scale, training_seed),
                scale=scale,
                training_seed=training_seed,
                calibration_seed=calibration_seed,
                evaluation_seed=evaluation_seed,
                launch_nonce=launch_nonce,
            ):
                _require(
                    not os.path.lexists(artifact_path),
                    f"Refusing calibration artifact created before child launch: "
                    f"{scale}/{training_seed}.",
                )
                gpu_lease.assert_held()
                device_guard.assert_held()
                with (
                    _held_retry_evidence_snapshots(admission),
                    _held_matrix_ledger_snapshot(
                        matrix_summary,
                        expected_payload=cast(Mapping[str, Any], current_prefix_payload),
                        trust_root=trust_root,
                    ),
                ):
                    result = _run_calibrator_from_stable_script(
                        command,
                        canonical=canonical_script,
                        expected=script_snapshot,
                        trust_root=trust_root,
                        gpu_lease=gpu_lease,
                        device_guard=device_guard,
                    )
                matrix_lock.assert_held()
                gpu_lease.assert_held()
                device_guard.assert_held()
                execution_environment.assert_exact_execution_environment(
                    frozen_execution_environment
                )
                training_matrix.assert_environment_unchanged(context)
                if not artifact_path.is_file():
                    if result.returncode not in {0, 2}:
                        raise subprocess.CalledProcessError(result.returncode, command)
                    raise ValueError(
                        f"Calibrator did not publish its exclusive artifact: "
                        f"{scale}/{training_seed}."
                    )
                published_sha256 = _sha256(artifact_path)
                published_bytes = artifact_path.stat().st_size
                artifact = load_and_validate_calibration_artifact(
                    artifact_path,
                    scale=scale,
                    training_seed=training_seed,
                    calibration_seed=calibration_seed,
                    evaluation_seed=evaluation_seed,
                    context=context,
                    training_context=training_context,
                    summary_path=summary_path,
                    summary=summary,
                    checkpoint=checkpoint,
                    command=command,
                    trust_root=trust_root,
                    training_matrix_summary_path=training_matrix_summary_path,
                    training_matrix_payload=training_matrix_payload,
                    ledger_record=ledger_records[(scale, training_seed)],
                )
                matrix_lock.assert_held()
                _require(
                    _sha256(artifact_path) == published_sha256
                    and artifact_path.stat().st_size == published_bytes,
                    "Exclusive calibration artifact changed during validation.",
                )
                _require(
                    result.returncode
                    == _expected_exit_code(cast(str, artifact["terminal_decision"])),
                    "Calibrator exit code does not match its MAC-attested terminal decision.",
                )
                _require(
                    artifact.get("environment") == frozen_execution_environment,
                    f"Calibration child execution environment drifted: {scale}/{training_seed}.",
                )
                new_record = _cell_record(artifact, artifact_path=artifact_path, command=command)

        if new_record is None:
            _require(
                artifact.get("environment") == frozen_execution_environment,
                f"Calibration child execution environment drifted: {scale}/{training_seed}.",
            )
            record = _cell_record(artifact, artifact_path=artifact_path, command=command)
        else:
            record = new_record
        matrix_lock.assert_held()
        if index < len(completed):
            _require(
                completed[index] == record,
                f"Previously completed calibration cell drifted: {scale}/{training_seed}.",
            )
        else:
            with _held_retry_evidence_snapshots(admission):
                with _held_matrix_ledger_snapshot(
                    matrix_summary,
                    expected_payload=cast(Mapping[str, Any], current_prefix_payload),
                    trust_root=trust_root,
                ):
                    candidate_completed = [*completed, record]
                    gpu_lease.assert_held()
                    if retry_admission_binding is not None:
                        _assert_retry_admission_file_binding(retry_admission_binding)
                    next_prefix_payload = _matrix_payload(
                        candidate_completed,
                        context=context,
                        calibration_script_binding=script_binding,
                        matrix_lock_binding=matrix_lock_binding,
                        gpu_lease_binding=gpu_lease_binding,
                        execution_environment_binding=frozen_execution_environment,
                        trust_root=trust_root,
                        retry_admission=retry_admission_binding,
                    )
                _publish_matrix_ledger(
                    matrix_summary,
                    next_prefix_payload,
                    matrix_lock=matrix_lock,
                    trust_root=trust_root,
                )
                completed.append(record)
                current_prefix_payload = next_prefix_payload
                gpu_lease.assert_held()

    gpu_lease.assert_held()
    training_matrix.assert_environment_unchanged(context)
    if retry_admission_binding is not None:
        _assert_retry_admission_file_binding(retry_admission_binding)
    terminal = _matrix_payload(
        completed,
        context=context,
        calibration_script_binding=script_binding,
        matrix_lock_binding=matrix_lock_binding,
        gpu_lease_binding=gpu_lease_binding,
        execution_environment_binding=frozen_execution_environment,
        trust_root=trust_root,
        retry_admission=retry_admission_binding,
    )
    gpu_lease.assert_held()
    _require(
        terminal == current_prefix_payload,
        "Terminal calibration payload differs from the last durably published prefix.",
    )
    with (
        _held_retry_evidence_snapshots(admission),
        _held_matrix_ledger_snapshot(
            matrix_summary,
            expected_payload=terminal,
            trust_root=trust_root,
        ),
    ):
        validate_matrix_summary(
            terminal,
            output_root=output_root,
            training_output_root=training_output_root,
            calibration_script=canonical_script,
            calibration_script_binding=script_binding,
            matrix_lock_binding=matrix_lock_binding,
            expected_gpu_lease=gpu_lease_binding,
            expected_execution_environment=frozen_execution_environment,
            context=context,
            training_context=training_context,
            trust_root=trust_root,
            training_matrix_summary_path=training_matrix_summary_path,
            training_matrix_payload=training_matrix_payload,
            trainer_binding=trainer_binding,
            ledger_records=ledger_records,
            retry_admission=retry_admission_binding,
        )
    matrix_lock.assert_held()
    gpu_lease.assert_held()
    device_guard.assert_held()
    return terminal


def _run_matrix_under_gpu_lease(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = TRAINING_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path | None = None,
    calibration_script: Path = CALIBRATION_SCRIPT,
    attestation_key_path: Path | None = None,
    gpu_lease: gpu_lock.GPULockLease,
    device_guard: gpu_lock.GPULockLease,
    gpu_lease_binding: Mapping[str, Any],
    frozen_execution_environment: Mapping[str, Any],
) -> dict[str, Any]:
    gpu_lease.assert_held()
    device_guard.assert_held()
    _require(
        dict(gpu_lease_binding)
        == _gpu_lease_binding(
            gpu_lease.path,
            frozen_execution_environment=frozen_execution_environment,
            device_guard=device_guard,
        ),
        "Calibration GPU lease binding does not match the held lease.",
    )
    _assert_frozen_contract()
    key_path_for_layout = attestation_key_path
    if key_path_for_layout is None:
        raw_key_path = os.environ.get(attestation.KEY_PATH_ENV)
        if raw_key_path:
            key_path_for_layout = Path(raw_key_path)
    requested_summary = (
        output_root / MATRIX_SUMMARY_NAME if matrix_summary is None else matrix_summary
    )
    layout = _validate_matrix_layout(
        output_root=output_root,
        matrix_summary=requested_summary,
        training_output_root=training_output_root,
        attestation_key_path=key_path_for_layout,
    )
    canonical_script = _canonical_calibration_script(calibration_script)
    lock_binding = _matrix_lock_binding(layout.lock_path)
    with _exclusive_matrix_lock(
        layout.lock_path, matrix_summary=layout.matrix_summary
    ) as matrix_lock:
        matrix_lock.assert_held()
        gpu_lease.assert_held()
        device_guard.assert_held()
        return _run_matrix_locked(
            manifest_path=manifest_path,
            training_output_root=layout.training_output_root,
            output_root=layout.output_root,
            matrix_summary=layout.matrix_summary,
            calibration_script=canonical_script,
            matrix_lock_binding=lock_binding,
            matrix_lock=matrix_lock,
            gpu_lease=gpu_lease,
            device_guard=device_guard,
            gpu_lease_binding=gpu_lease_binding,
            frozen_execution_environment=frozen_execution_environment,
            attestation_key_path=attestation_key_path,
        )


def run_matrix(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = TRAINING_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path | None = None,
    calibration_script: Path = CALIBRATION_SCRIPT,
    attestation_key_path: Path | None = None,
    gpu_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Run the complete CUDA calibration matrix under one project-wide GPU lease."""

    requested_gpu_lock = (
        RETRY_GPU_LOCK_PATH
        if REQUIRE_RETRY_ADMISSION and gpu_lock_path is None
        else gpu_lock.DEFAULT_LOCK_PATH
        if gpu_lock_path is None
        else gpu_lock_path
    )
    if REQUIRE_RETRY_ADMISSION:
        requested_lock = _absolute_path(requested_gpu_lock).resolve(strict=False)
        superseded_root = _absolute_path(SUPERSEDED_OUTPUT_ROOT).resolve(strict=False)
        _require(
            requested_lock == _absolute_path(RETRY_GPU_LOCK_PATH).resolve(strict=False)
            and not _paths_overlap(requested_lock, superseded_root),
            "Revision 1.2 retry requires the frozen failed-attempt GPU scheduler lock outside "
            "the quarantine.",
        )
    lease = gpu_lock.acquire_gpu_lock(
        "p2-direct-calibration-matrix",
        path=requested_gpu_lock,
    )
    device_guard: gpu_lock.GPULockLease | None = None
    try:
        lease.assert_held()
        frozen_execution_environment = execution_environment.capture_execution_environment()
        lease.assert_held()
        routing_identity = execution_environment.selected_device_routing_identity(
            frozen_execution_environment
        )
        device_guard = gpu_lock.acquire_device_guard(
            "p2-direct-calibration-matrix",
            routing_identity,
        )
        device_guard.assert_held()
        lease_binding = _gpu_lease_binding(
            lease.path,
            frozen_execution_environment=frozen_execution_environment,
            device_guard=device_guard,
        )
        return _run_matrix_under_gpu_lease(
            manifest_path=manifest_path,
            training_output_root=training_output_root,
            output_root=output_root,
            matrix_summary=matrix_summary,
            calibration_script=calibration_script,
            attestation_key_path=attestation_key_path,
            gpu_lease=lease,
            device_guard=device_guard,
            gpu_lease_binding=lease_binding,
            frozen_execution_environment=frozen_execution_environment,
        )
    finally:
        if device_guard is not None:
            device_guard.close()
        lease.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume-safe launcher for the frozen post-rank direct SoftLag calibration grid."
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--training-output-root", type=Path, default=TRAINING_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--matrix-summary", type=Path)
    parser.add_argument("--gpu-lock-path", type=Path)
    args = parser.parse_args()
    result = run_matrix(
        manifest_path=args.manifest,
        training_output_root=args.training_output_root,
        output_root=args.output_root,
        matrix_summary=(
            args.output_root / MATRIX_SUMMARY_NAME
            if args.matrix_summary is None
            else args.matrix_summary
        ),
        calibration_script=CALIBRATION_SCRIPT,
        gpu_lock_path=args.gpu_lock_path,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "terminal_decision": result["terminal_decision"],
                "completed_cells": result["completed_cells"],
                "quality_evaluation_started": result["quality_evaluation_started"],
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return _expected_exit_code(cast(str, result["terminal_decision"]))


if __name__ == "__main__":
    raise SystemExit(main())
