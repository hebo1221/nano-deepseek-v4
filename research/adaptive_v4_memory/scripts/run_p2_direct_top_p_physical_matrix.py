from __future__ import annotations

import argparse
import fcntl
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
import calibrate_p2_direct_soft_lag as calibration_program
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import run_p2_direct_calibration_matrix as calibration_matrix
import run_p2_direct_training_matrix as training_matrix
import validate_p2_direct_top_p_physical_match as physical_match
from adaptive_v4_gpu_lock import (
    DEVICE_GUARD_SEMANTICS,
    GPULockLease,
    acquire_device_guard,
    acquire_gpu_lock,
    canonical_device_guard_path,
)

EXPERIMENT_ID = "p2-post-rank-direct-top-p-physical-matrix-v1"
ARTIFACT_TYPE = "direct-top-p-physical-match-matrix"
SCHEMA_VERSION = 5
MATRIX_ATTESTATION_PURPOSE = "p2-direct-top-p-physical-match-matrix-v1"

FROZEN_SCALES = tuple(contract.SCALES)
FROZEN_TRAINING_SEEDS = tuple(contract.TRAINING_SEEDS)
FROZEN_BUDGETS = tuple(contract.BUDGETS)
FROZEN_COMPARATORS = tuple(contract.SENSITIVITY_COMPARATOR_ARMS)
EXPECTED_CELLS = (
    len(FROZEN_SCALES) * len(FROZEN_TRAINING_SEEDS) * len(FROZEN_BUDGETS) * len(FROZEN_COMPARATORS)
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
GENERATOR_SCRIPT = Path(__file__).resolve().with_name("validate_p2_direct_top_p_physical_match.py")
GENERATOR_IMPLEMENTATION_PATH = GENERATOR_SCRIPT.relative_to(REPOSITORY_ROOT).as_posix()
RUNNER_IMPLEMENTATION_PATH = Path(__file__).relative_to(REPOSITORY_ROOT).as_posix()
GPU_LOCK_IMPLEMENTATION_PATH = "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py"
GENERATOR_FD_ENV = "ADAPTIVE_V4_CANONICAL_TOP_P_MATCH_GENERATOR_FD"
GENERATOR_FD_BOOTSTRAP = (
    "import os,sys;"
    "p=sys.argv.pop(1);"
    "sys.path.insert(0,os.path.dirname(p));"
    f"fd=int(os.environ[{GENERATOR_FD_ENV!r}]);"
    "data=b'';"
    "\nwhile True:"
    "\n chunk=os.read(fd,1048576)"
    "\n if not chunk: break"
    "\n data+=chunk"
    "\ng={'__name__':'__main__','__file__':p,'__package__':None};"
    "exec(compile(data,p,'exec'),g,g)"
)

TRAINING_OUTPUT_ROOT = training_matrix.OUTPUT_ROOT
CALIBRATION_OUTPUT_ROOT = calibration_matrix.OUTPUT_ROOT
OUTPUT_ROOT = physical_match.DEFAULT_OUTPUT_ROOT
MATRIX_SUMMARY_NAME = "top-p-physical-matrix.summary.json"
MATRIX_SUMMARY = OUTPUT_ROOT / MATRIX_SUMMARY_NAME
MATRIX_LOCK_SUFFIX = "p2-direct-top-p-physical-matrix.lock"
MATRIX_LOCK_SEMANTICS = "persistent-sibling-flock-exclusive-process-owner-v1"
GPU_EXECUTION_LEASE_SEMANTICS = "project-persistent-inode-exclusive-whole-matrix-v1"
GPU_EXECUTION_LEASE_SCOPE = "before-preflight-through-terminal-validation"
GPU_DEVICE_GUARD_SCOPE = "after-exact-environment-capture-through-terminal-validation"
CELL_CLAIM_NAME = ".p2-direct-top-p-physical-cell.claim"
CELL_CLAIM_SEMANTICS = "exclusive-create-coordinate-nonce-preserve-on-failure-v2"
DEVICE = "cuda"
CRASH_RECOVERY_BOUNDARY = (
    "fail-closed: a claim or physical-match artifact outside the MAC-attested completed "
    "matrix prefix is orphan evidence and is never auto-promoted; quarantine is required"
)
OUTCOME_POLICY = (
    "execute all 40 frozen coordinates regardless of GO/NO-GO outcomes; only structural "
    "integrity or an explicit max-new-cells operational boundary may stop publication"
)
EXECUTION_THREAT_BOUNDARY = (
    "controlled-single-owner-host: Python, PyTorch, CUDA driver/runtime, and visible-device "
    "identity are frozen and replayed exactly; canonical generator bytes and the attestation "
    "key use sealed file descriptors, but imported modules and the process runtime remain live. "
    "The manifest-bound implementation tree is required clean before and after each launch; "
    "transient same-owner substitution entirely between those checks is outside this contract."
)
VALIDATION_API_SEMANTICS = (
    "three-explicit-replay-modes-v2: validate_matrix_summary performs exact same-host replay under "
    "an owned or borrowed GPU lease; validate_matrix_summary_controller_compatible requires a "
    "caller-held lease and relaxes CUDA routing topology while preserving the exact selected-device "
    "hardware and software class; validate_matrix_summary_archived validates stored HMAC, schema, "
    "artifacts, and closed-world coverage without a live CUDA probe or GPU lease"
)
SAFE_FILE_MODE = 0o600

_MATRIX_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "terminal_decision",
        "source",
        "manifest",
        "attestation_contract",
        "prerequisites",
        "canonical_generator",
        "execution_environment",
        "gpu_execution_lease",
        "execution_threat_boundary",
        "validation_api_semantics",
        "matrix_lock",
        "cell_claim_semantics",
        "crash_recovery_boundary",
        "outcome_policy",
        "outcome_dependent_early_stopping",
        "outcome_selection_performed",
        "coordinate_order",
        "coordinate_count",
        "coordinate_digest",
        "completed_cells",
        "go_cells",
        "no_go_cells",
        "records",
        "payload_sha256",
        "attestation",
    }
)
_EXECUTION_ENVIRONMENT_FIELDS = execution_environment.EXECUTION_ENVIRONMENT_FIELDS
_VISIBLE_DEVICE_FIELDS = execution_environment.VISIBLE_DEVICE_FIELDS
_GENERATOR_COMPLETION_FIELDS = frozenset(
    {
        "artifact",
        "mode",
        "resumed_existing_terminal_artifact",
        "terminal_decision",
        "payload_sha256",
        "python",
        "torch",
        "device_spec",
        "logical_device_index",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "scale",
        "training_seed",
        "calibration_seed",
        "budget",
        "comparator",
        "coordinate_key",
        "launch_nonce",
        "status",
        "terminal_decision",
        "generator_exit_code",
        "generator_completion",
        "command",
        "calibration_input",
        "artifact",
    }
)
_FILE_BINDING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "bytes",
        "payload_sha256",
        "attestation_mac",
        "experiment_id",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _assert_opened_file_security(opened: attestation.OpenedRegularFile, *, label: str) -> None:
    descriptor_metadata = os.fstat(opened.file_descriptor)
    try:
        path_metadata = os.stat(opened.path, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} disappeared during security validation.") from error
    _require(
        (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        == (path_metadata.st_dev, path_metadata.st_ino),
        f"{label} path was replaced during security validation.",
    )
    _require(
        stat.S_ISREG(descriptor_metadata.st_mode)
        and stat.S_ISREG(path_metadata.st_mode)
        and descriptor_metadata.st_uid == path_metadata.st_uid == os.getuid()
        and descriptor_metadata.st_nlink == path_metadata.st_nlink == 1
        and stat.S_IMODE(descriptor_metadata.st_mode)
        == stat.S_IMODE(path_metadata.st_mode)
        == SAFE_FILE_MODE,
        f"{label} ownership, links, or mode are unsafe.",
    )


def _assert_secure_regular_file(path: Path, *, label: str) -> None:
    opened = attestation.open_regular_nofollow(path)
    try:
        _assert_opened_file_security(opened, label=label)
        opened.assert_unchanged()
        _assert_opened_file_security(opened, label=label)
    finally:
        opened.close()


def _query_cuda_driver_version() -> str:
    return execution_environment.query_cuda_driver_version()


def _validate_execution_environment(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return execution_environment.validate_execution_environment(payload)
    except ValueError as error:
        raise ValueError(f"Top-p {error}") from error


def _capture_execution_environment() -> dict[str, Any]:
    return execution_environment.capture_execution_environment()


def _assert_execution_environment_unchanged(expected: Mapping[str, Any]) -> None:
    frozen = _validate_execution_environment(expected)
    current = _capture_execution_environment()
    _require(current == frozen, "Top-p execution environment changed after freeze.")


def _capture_execution_environment_under_lease(gpu_lease: GPULockLease) -> dict[str, Any]:
    gpu_lease.assert_held()
    snapshot = _capture_execution_environment()
    gpu_lease.assert_held()
    return snapshot


def _assert_device_guard_matches_environment(
    device_guard_lease: GPULockLease,
    current_execution_environment: Mapping[str, Any],
) -> None:
    device_guard_lease.assert_held()
    routing_identity = execution_environment.selected_device_routing_identity(
        current_execution_environment
    )
    _require(
        device_guard_lease.path == canonical_device_guard_path(routing_identity),
        "Top-p physical-device guard is not bound to the current selected GPU.",
    )


def _assert_execution_environment_unchanged_under_lease(
    expected: Mapping[str, Any], *, gpu_lease: GPULockLease
) -> None:
    gpu_lease.assert_held()
    _assert_execution_environment_unchanged(expected)
    gpu_lease.assert_held()


def _controller_compatible_environment_projection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return execution_environment.controller_compatible_environment_projection(payload)


def _assert_controller_compatible_execution_environment(
    expected: Mapping[str, Any], *, gpu_lease: GPULockLease
) -> None:
    gpu_lease.assert_held()
    current = _capture_execution_environment()
    gpu_lease.assert_held()
    _require(
        _controller_compatible_environment_projection(current)
        == _controller_compatible_environment_projection(expected),
        "Top-p execution environment is not controller-compatible with the frozen hardware class.",
    )


@dataclass(frozen=True)
class MatchCoordinate:
    scale: str
    training_seed: int
    calibration_seed: int
    budget: str
    comparator: str

    @property
    def key(self) -> str:
        return f"{self.scale}/train-{self.training_seed}/{self.budget}/{self.comparator}"

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "training_seed": self.training_seed,
            "calibration_seed": self.calibration_seed,
            "budget": self.budget,
            "comparator": self.comparator,
        }


@dataclass(frozen=True)
class CanonicalGeneratorSnapshot:
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
            "implementation_path": GENERATOR_IMPLEMENTATION_PATH,
        }


@dataclass(frozen=True)
class CalibrationInput:
    coordinate: tuple[str, int]
    path: Path
    payload: Mapping[str, Any]
    binding: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenPrerequisites:
    context: training_matrix.FrozenContext
    trust_root: attestation.TrustRoot
    calibrations: Mapping[tuple[str, int], CalibrationInput]
    public_binding: Mapping[str, Any]


@dataclass(frozen=True)
class MatrixLayout:
    output_root: Path
    matrix_summary: Path
    lock_path: Path
    training_output_root: Path
    calibration_output_root: Path


_ACTIVE_MATRIX_LOCKS: set[str] = set()
_ACTIVE_MATRIX_LOCKS_GUARD = threading.Lock()


def coordinates() -> tuple[MatchCoordinate, ...]:
    result: list[MatchCoordinate] = []
    for scale in FROZEN_SCALES:
        for training_seed in FROZEN_TRAINING_SEEDS:
            aligned, calibration_seed, _evaluation_seed = contract.seed_triplet(training_seed)
            _require(aligned == training_seed, "Frozen top-p seed alignment drifted.")
            for budget in FROZEN_BUDGETS:
                for comparator in FROZEN_COMPARATORS:
                    result.append(
                        MatchCoordinate(
                            scale=scale,
                            training_seed=training_seed,
                            calibration_seed=calibration_seed,
                            budget=budget,
                            comparator=comparator,
                        )
                    )
    _require(len(result) == EXPECTED_CELLS == 40, "Top-p matrix is not exactly 40 cells.")
    _require(len({item.key for item in result}) == len(result), "Top-p grid repeats a cell.")
    return tuple(result)


def coordinate_digest() -> str:
    return contract.json_digest([item.payload for item in coordinates()])


def artifact_path(output_root: Path, coordinate: MatchCoordinate) -> Path:
    return physical_match.artifact_path(
        output_root,
        coordinate.scale,
        coordinate.training_seed,
        coordinate.budget,
        coordinate.comparator,
    )


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _exact_resolved_path(path: Path, *, label: str) -> Path:
    absolute = _absolute(path)
    _require(not absolute.is_symlink(), f"{label} may not be a symbolic link.")
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as error:
        raise ValueError(f"{label} cannot be resolved safely.") from error
    _require(resolved == absolute, f"{label} must use its exact resolved path.")
    return absolute


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _gpu_execution_lease_binding(
    gpu_lease: GPULockLease,
    *,
    device_guard_lease: GPULockLease,
    current_execution_environment: Mapping[str, Any],
) -> dict[str, Any]:
    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    path = _exact_resolved_path(gpu_lease.path, label="Top-p GPU execution lease")
    routing_identity = execution_environment.selected_device_routing_identity(
        current_execution_environment
    )
    guard_path = canonical_device_guard_path(routing_identity)
    _require(
        device_guard_lease.path == guard_path,
        "Top-p physical-device guard does not match the selected GPU.",
    )
    return {
        "path": str(path),
        "semantics": GPU_EXECUTION_LEASE_SEMANTICS,
        "scope": GPU_EXECUTION_LEASE_SCOPE,
        "implementation_path": GPU_LOCK_IMPLEMENTATION_PATH,
        "nonblocking": True,
        "persistent_inode": True,
        "st_dev": gpu_lease.device,
        "st_ino": gpu_lease.inode,
        "selected_device_class": execution_environment.selected_device_class(
            current_execution_environment
        ),
        "selected_device_routing_identity": routing_identity,
        "device_guard": {
            "path": str(guard_path),
            "semantics": DEVICE_GUARD_SEMANTICS,
            "scope": GPU_DEVICE_GUARD_SCOPE,
            "implementation_path": GPU_LOCK_IMPLEMENTATION_PATH,
            "nonblocking": True,
            "persistent_inode": True,
            "st_dev": device_guard_lease.device,
            "st_ino": device_guard_lease.inode,
        },
    }


def _validate_gpu_execution_lease_binding(
    value: Any,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "Top-p GPU execution lease binding is missing.")
    binding = cast(Mapping[str, Any], value)
    _require(
        set(binding)
        == {
            "path",
            "semantics",
            "scope",
            "implementation_path",
            "nonblocking",
            "persistent_inode",
            "st_dev",
            "st_ino",
            "selected_device_class",
            "selected_device_routing_identity",
            "device_guard",
        },
        "Top-p GPU execution lease schema drifted.",
    )
    raw_path = binding.get("path")
    _require(
        isinstance(raw_path, str) and bool(raw_path),
        "Top-p GPU execution lease path is invalid.",
    )
    path = Path(cast(str, raw_path))
    _require(
        path == Path(os.path.abspath(path)),
        "Top-p GPU execution lease path is not lexically canonical.",
    )
    replayed = {
        "path": str(path),
        "semantics": GPU_EXECUTION_LEASE_SEMANTICS,
        "scope": GPU_EXECUTION_LEASE_SCOPE,
        "implementation_path": GPU_LOCK_IMPLEMENTATION_PATH,
        "nonblocking": True,
        "persistent_inode": True,
        "st_dev": binding.get("st_dev"),
        "st_ino": binding.get("st_ino"),
        "selected_device_class": binding.get("selected_device_class"),
        "selected_device_routing_identity": binding.get("selected_device_routing_identity"),
        "device_guard": binding.get("device_guard"),
    }
    _require(
        type(replayed["st_dev"]) is int
        and replayed["st_dev"] >= 0
        and type(replayed["st_ino"]) is int
        and replayed["st_ino"] > 0,
        "Top-p GPU execution lease inode identity is invalid.",
    )
    selected_class = replayed["selected_device_class"]
    routing_identity = replayed["selected_device_routing_identity"]
    guard = replayed["device_guard"]
    _require(
        isinstance(selected_class, Mapping)
        and set(selected_class) == execution_environment.SELECTED_DEVICE_CLASS_FIELDS
        and isinstance(routing_identity, Mapping)
        and set(routing_identity) == execution_environment.SELECTED_DEVICE_ROUTING_IDENTITY_FIELDS
        and isinstance(guard, Mapping),
        "Top-p selected-device lease binding schema drifted.",
    )
    expected_guard_path = canonical_device_guard_path(cast(Mapping[str, Any], routing_identity))
    guard = cast(Mapping[str, Any], guard)
    _require(
        set(guard)
        == {
            "path",
            "semantics",
            "scope",
            "implementation_path",
            "nonblocking",
            "persistent_inode",
            "st_dev",
            "st_ino",
        }
        and guard.get("path") == str(expected_guard_path)
        and guard.get("semantics") == DEVICE_GUARD_SEMANTICS
        and guard.get("scope") == GPU_DEVICE_GUARD_SCOPE
        and guard.get("implementation_path") == GPU_LOCK_IMPLEMENTATION_PATH
        and guard.get("nonblocking") is True
        and guard.get("persistent_inode") is True
        and type(guard.get("st_dev")) is int
        and cast(int, guard["st_dev"]) >= 0
        and type(guard.get("st_ino")) is int
        and cast(int, guard["st_ino"]) > 0,
        "Top-p physical-device guard binding drifted.",
    )
    _require(dict(binding) == replayed, "Top-p GPU execution lease semantics drifted.")
    if expected is not None:
        _require(
            replayed == dict(expected),
            "Top-p GPU execution lease path or inode drifted on same-host replay.",
        )
    return replayed


def _matrix_lock_path(output_root: Path) -> Path:
    root = _absolute(output_root)
    _require(root != root.parent and bool(root.name), "Top-p output root may not be a root.")
    return root.parent / f".{root.name}.{MATRIX_LOCK_SUFFIX}"


def _validate_matrix_layout(
    *,
    output_root: Path,
    matrix_summary: Path,
    training_output_root: Path,
    calibration_output_root: Path,
    attestation_key_path: Path | None,
) -> MatrixLayout:
    root = _exact_resolved_path(output_root, label="Top-p output root")
    summary = _exact_resolved_path(matrix_summary, label="Top-p matrix summary")
    _require(
        summary == root / MATRIX_SUMMARY_NAME,
        "Top-p matrix summary must use its canonical output-root path.",
    )
    training_root = _exact_resolved_path(training_output_root, label="Training input root")
    calibration_root = _exact_resolved_path(calibration_output_root, label="Calibration input root")
    superseded_calibration_root = _absolute(
        calibration_matrix.SUPERSEDED_OUTPUT_ROOT
    ).resolve(strict=False)
    _require(
        all(
            not _paths_overlap(item, superseded_calibration_root)
            for item in (root, training_root, calibration_root)
        ),
        "Top-p paths may not overlap the immutable revision 1.1 calibration quarantine.",
    )
    _require(
        not _paths_overlap(training_root, calibration_root),
        "Training and calibration input roots must be disjoint.",
    )
    _require(
        not _paths_overlap(root, training_root) and not _paths_overlap(root, calibration_root),
        "Top-p output root must be disjoint from prerequisite roots.",
    )
    lock_path = _exact_resolved_path(_matrix_lock_path(root), label="Top-p matrix lock")
    _require(
        not _paths_overlap(lock_path, root)
        and not _paths_overlap(lock_path, training_root)
        and not _paths_overlap(lock_path, calibration_root),
        "Top-p matrix lock must be a disjoint sibling path.",
    )
    _require(
        not _paths_overlap(lock_path, superseded_calibration_root),
        "Top-p matrix lock may not overlap the immutable calibration quarantine.",
    )
    if attestation_key_path is not None:
        key = _exact_resolved_path(attestation_key_path, label="Attestation key")
        _require(
            not key.is_relative_to(REPOSITORY_ROOT.resolve(strict=True)),
            "Attestation key must be outside the repository.",
        )
        _require(
            key != lock_path
            and not _paths_overlap(key, root)
            and not _paths_overlap(key, training_root)
            and not _paths_overlap(key, calibration_root),
            "Attestation key must be disjoint from matrix paths.",
        )
    paths = [artifact_path(root, coordinate) for coordinate in coordinates()]
    _require(len(set(paths)) == EXPECTED_CELLS, "Top-p artifact paths collide.")
    _require(summary not in paths, "Top-p summary collides with a physical artifact.")
    return MatrixLayout(
        output_root=root,
        matrix_summary=summary,
        lock_path=lock_path,
        training_output_root=training_root,
        calibration_output_root=calibration_root,
    )


def _write_fd_json(file_descriptor: int, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    os.ftruncate(file_descriptor, 0)
    offset = 0
    while offset < len(encoded):
        written = os.write(file_descriptor, encoded[offset:])
        _require(written > 0, "Lock metadata write stalled.")
        offset += written
    os.fsync(file_descriptor)


def _matrix_lock_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "semantics": MATRIX_LOCK_SEMANTICS,
        "persistent_inode": True,
        "unlink_on_release": False,
    }


def _assert_lock_identity(
    descriptor: int,
    lock_path: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> None:
    opened = os.fstat(descriptor)
    try:
        current = os.stat(lock_path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("Top-p matrix lock path disappeared while held.") from error
    expected_identity = (expected_device, expected_inode)
    _require(
        (opened.st_dev, opened.st_ino) == expected_identity
        and (current.st_dev, current.st_ino) == expected_identity,
        "Top-p matrix lock path was replaced while held.",
    )
    _require(
        stat.S_ISREG(opened.st_mode)
        and stat.S_ISREG(current.st_mode)
        and opened.st_uid == current.st_uid == os.getuid()
        and opened.st_nlink == current.st_nlink == 1
        and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(current.st_mode) == SAFE_FILE_MODE,
        "Top-p matrix lock ownership, links, or mode changed while held.",
    )


@contextmanager
def _exclusive_matrix_lock(
    lock_path: Path, *, matrix_summary: Path
) -> Iterator[Callable[[], None]]:
    key = str(lock_path)
    with _ACTIVE_MATRIX_LOCKS_GUARD:
        _require(key not in _ACTIVE_MATRIX_LOCKS, "Top-p matrix lock reentry is forbidden.")
        _ACTIVE_MATRIX_LOCKS.add(key)
    descriptor: int | None = None
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
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        no_follow = getattr(os, "O_NOFOLLOW", None)
        _require(no_follow is not None, "Top-p matrix locking requires O_NOFOLLOW.")
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
                SAFE_FILE_MODE,
            )
        except OSError as error:
            raise ValueError("Top-p matrix lock could not be opened safely.") from error
        opened = os.fstat(descriptor)
        current = os.stat(lock_path, follow_symlinks=False)
        _require(stat.S_ISREG(opened.st_mode), "Top-p matrix lock is not regular.")
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Top-p matrix lock path changed while opening.",
        )
        _require(
            opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == SAFE_FILE_MODE,
            "Top-p matrix lock ownership, links, or mode are unsafe.",
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = True
        _assert_lock_identity(
            descriptor,
            lock_path,
            expected_device=opened.st_dev,
            expected_inode=opened.st_ino,
        )
        _write_fd_json(descriptor, owner)
        _assert_lock_identity(
            descriptor,
            lock_path,
            expected_device=opened.st_dev,
            expected_inode=opened.st_ino,
        )

        def assert_held() -> None:
            _assert_lock_identity(
                descriptor,
                lock_path,
                expected_device=opened.st_dev,
                expected_inode=opened.st_ino,
            )

        yield assert_held
        _assert_lock_identity(
            descriptor,
            lock_path,
            expected_device=opened.st_dev,
            expected_inode=opened.st_ino,
        )
        _write_fd_json(
            descriptor,
            {**owner, "state": "released", "released_time_ns": time.time_ns()},
        )
        _assert_lock_identity(
            descriptor,
            lock_path,
            expected_device=opened.st_dev,
            expected_inode=opened.st_ino,
        )
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        acquired = False
    finally:
        if descriptor is not None:
            if acquired:
                try:
                    _assert_lock_identity(
                        descriptor,
                        lock_path,
                        expected_device=opened.st_dev,
                        expected_inode=opened.st_ino,
                    )
                    _write_fd_json(
                        descriptor,
                        {
                            **owner,
                            "state": "released-after-error",
                            "released_time_ns": time.time_ns(),
                        },
                    )
                    _assert_lock_identity(
                        descriptor,
                        lock_path,
                        expected_device=opened.st_dev,
                        expected_inode=opened.st_ino,
                    )
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
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
        raise ValueError("Top-p cell claim disappeared while held.") from error
    expected_identity = (expected_device, expected_inode)
    _require(
        (opened.st_dev, opened.st_ino) == expected_identity
        and (current.st_dev, current.st_ino) == expected_identity,
        "Top-p cell claim path was replaced while held.",
    )
    _require(
        stat.S_ISREG(opened.st_mode)
        and stat.S_ISREG(current.st_mode)
        and opened.st_uid == current.st_uid == os.getuid()
        and opened.st_nlink == current.st_nlink == 1
        and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(current.st_mode) == SAFE_FILE_MODE,
        "Top-p cell claim ownership, links, or mode changed while held.",
    )


@contextmanager
def _exclusive_cell_claim(
    output_dir: Path, *, coordinate: MatchCoordinate, launch_nonce: str
) -> Iterator[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / CELL_CLAIM_NAME
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Top-p cell claims require O_NOFOLLOW.")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
    )
    try:
        descriptor = os.open(claim_path, flags, SAFE_FILE_MODE)
    except FileExistsError as error:
        raise ValueError(f"Top-p cell has an orphan or active claim: {coordinate.key}") from error
    metadata: os.stat_result | None = None
    release_claim = False
    payload = {
        "schema_version": 1,
        "semantics": CELL_CLAIM_SEMANTICS,
        "coordinate": coordinate.payload,
        "launch_nonce": launch_nonce,
        "pid": os.getpid(),
        "created_time_ns": time.time_ns(),
    }
    try:
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE
            and metadata.st_nlink == 1,
            "Top-p cell claim metadata is unsafe.",
        )
        _write_fd_json(descriptor, payload)
        _assert_cell_claim_identity(
            descriptor,
            claim_path,
            expected_device=metadata.st_dev,
            expected_inode=metadata.st_ino,
        )
        yield {"path": str(claim_path), "semantics": CELL_CLAIM_SEMANTICS}
        release_claim = True
    finally:
        try:
            if metadata is None:
                raise ValueError("Top-p cell claim metadata was not established.")
            _assert_cell_claim_identity(
                descriptor,
                claim_path,
                expected_device=metadata.st_dev,
                expected_inode=metadata.st_ino,
            )
            if release_claim:
                claim_path.unlink()
                _require(
                    os.fstat(descriptor).st_nlink == 0 and not _path_lexists(claim_path),
                    "Top-p cell claim unlink target drifted.",
                )
        finally:
            os.close(descriptor)


def _load_json_nofollow(path: Path, *, label: str) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        _assert_opened_file_security(opened, label=label)
        try:
            payload = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid {label}: {path}") from error
        _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
        opened.assert_unchanged()
        _assert_opened_file_security(opened, label=label)
        return payload
    finally:
        opened.close()


def _assert_exact_json_payload_file(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    opened = attestation.open_regular_nofollow(path)
    try:
        _assert_opened_file_security(opened, label=label)
        expected = (
            json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        _require(
            opened.read_bytes() == expected,
            f"{label} is not the exact canonical published byte encoding.",
        )
        opened.assert_unchanged()
        _assert_opened_file_security(opened, label=label)
    finally:
        opened.close()


def _file_binding(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        _assert_opened_file_security(opened, label="Attested file")
        try:
            parsed = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Attested file binding is not valid JSON: {path}") from error
        _require(isinstance(parsed, dict), f"Attested file binding is not an object: {path}")
        _require(
            attestation.canonical_json(parsed) == attestation.canonical_json(dict(payload)),
            f"Attested file bytes do not match the supplied payload: {path}",
        )
        envelope = parsed.get("attestation")
        _require(isinstance(envelope, Mapping), f"Attestation is missing from {path}.")
        binding = {
            "path": str(opened.path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
            "payload_sha256": parsed.get("payload_sha256"),
            "attestation_mac": cast(Mapping[str, Any], envelope).get("mac"),
            "experiment_id": parsed.get("experiment_id"),
        }
        _require(
            contract.is_sha256(binding["payload_sha256"])
            and contract.is_sha256(binding["attestation_mac"]),
            f"Attested file binding is incomplete: {path}",
        )
        opened.assert_unchanged()
        _assert_opened_file_security(opened, label="Attested file")
        return binding
    finally:
        opened.close()


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
            os.fchmod(handle.fileno(), SAFE_FILE_MODE)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _assert_secure_regular_file(path, label="Published top-p matrix ledger")
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _digest_bound_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("attestation", None)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = contract.json_digest(result)
    return result


def _attested_payload(
    payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    result = _digest_bound_payload(payload)
    result["attestation"] = attestation.attest_payload(
        result,
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
    )
    return result


def _verify_attested_payload(
    payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> None:
    _require(set(payload) == _MATRIX_FIELDS, "Top-p matrix schema drifted.")
    digest = payload.get("payload_sha256")
    digest_source = dict(payload)
    digest_source.pop("attestation", None)
    digest_source.pop("payload_sha256", None)
    _require(
        contract.is_sha256(digest) and digest == contract.json_digest(digest_source),
        "Top-p matrix payload digest drifted.",
    )
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Top-p matrix attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
    )


def _canonical_generator(candidate: Path) -> Path:
    _require(not candidate.is_symlink(), "Top-p generator may not be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
        canonical = GENERATOR_SCRIPT.resolve(strict=True)
    except OSError as error:
        raise ValueError("Canonical top-p generator is missing or inaccessible.") from error
    _require(resolved == canonical, "Top-p generator must be the canonical manifest-bound script.")
    _require(not canonical.is_symlink(), "Canonical top-p generator may not be a symlink.")
    return canonical


def _generator_snapshot(opened: attestation.OpenedRegularFile) -> CanonicalGeneratorSnapshot:
    return CanonicalGeneratorSnapshot(
        path=str(opened.path),
        sha256=opened.sha256,
        bytes=opened.bytes,
        device=opened.device,
        inode=opened.inode,
        mode=opened.mode,
        mtime_ns=opened.mtime_ns,
        ctime_ns=opened.ctime_ns,
    )


def _open_generator(
    canonical: Path, *, expected: CanonicalGeneratorSnapshot | None = None
) -> tuple[attestation.OpenedRegularFile, CanonicalGeneratorSnapshot]:
    opened = attestation.open_regular_nofollow(canonical)
    try:
        snapshot = _generator_snapshot(opened)
        if expected is not None:
            _require(snapshot == expected, "Canonical top-p generator changed after preflight.")
        return opened, snapshot
    except BaseException:
        opened.close()
        raise


def _sealed_generator_copy(
    opened: attestation.OpenedRegularFile, *, expected: CanonicalGeneratorSnapshot
) -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(
        create is not None and allow_sealing is not None,
        "Stable top-p execution requires sealed memfd support.",
    )
    sealed = cast(Callable[[str, int], int], create)(
        "adaptive-v4-canonical-top-p-generator",
        cast(int, getattr(os, "MFD_CLOEXEC", 0)) | cast(int, allow_sealing),
    )
    try:
        payload = opened.read_bytes()
        _require(
            len(payload) == expected.bytes
            and attestation.checksum_bytes(payload) == expected.sha256,
            "Canonical top-p generator bytes drifted during snapshotting.",
        )
        written = 0
        while written < len(payload):
            count = os.write(sealed, payload[written:])
            _require(count > 0, "Sealed top-p generator write stalled.")
            written += count
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(sealed, fcntl.F_ADD_SEALS, seals)
        _require(
            fcntl.fcntl(sealed, fcntl.F_GET_SEALS) & seals == seals,
            "Top-p generator memfd was not sealed.",
        )
        os.lseek(sealed, 0, os.SEEK_SET)
        return sealed
    except BaseException:
        os.close(sealed)
        raise


def _run_generator_from_snapshot(
    command: Sequence[str],
    *,
    canonical: Path,
    expected: CanonicalGeneratorSnapshot,
    trust_root: attestation.TrustRoot,
    gpu_lease: GPULockLease,
    device_guard_lease: GPULockLease,
) -> subprocess.CompletedProcess[Any]:
    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    opened, snapshot = _open_generator(canonical, expected=expected)
    sealed: int | None = None
    key_descriptor: int | None = None
    try:
        sealed = _sealed_generator_copy(opened, expected=snapshot)
        key_descriptor = attestation.create_sealed_key_fd(trust_root)
        environment = os.environ.copy()
        environment.pop(attestation.KEY_PATH_ENV, None)
        environment[GENERATOR_FD_ENV] = str(sealed)
        environment[attestation.KEY_FD_ENV] = str(key_descriptor)
        result = subprocess.run(
            list(command),
            check=False,
            pass_fds=(
                sealed,
                key_descriptor,
                gpu_lease.fileno(),
                device_guard_lease.fileno(),
            ),
            env=environment,
            capture_output=True,
            text=True,
        )
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
        opened.assert_unchanged()
        return result
    finally:
        if key_descriptor is not None:
            os.close(key_descriptor)
        if sealed is not None:
            os.close(sealed)
        opened.close()


def _assert_implementation_binding(context: training_matrix.FrozenContext) -> None:
    _require(
        GENERATOR_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS,
        "Canonical top-p generator is absent from the implementation inventory.",
    )
    _require(
        RUNNER_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS,
        "Top-p matrix runner is absent from the implementation inventory.",
    )
    _require(
        GPU_LOCK_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS,
        "Project-wide GPU lock is absent from the implementation inventory.",
    )
    _require(
        contract.implementation_tree_digest()
        == context.manifest_binding.get("implementation_digest"),
        "Top-p matrix implementation is not bound by the frozen manifest.",
    )


def load_and_validate_prerequisites(
    *,
    manifest_path: Path,
    training_output_root: Path,
    calibration_output_root: Path,
    output_root: Path,
    attestation_key_path: Path | None,
) -> FrozenPrerequisites:
    """Authenticate the terminal training/calibration chain before any CUDA launch."""

    context = training_matrix.establish_frozen_context(manifest_path)
    raw_attestation = context.manifest_binding.get("attestation")
    _require(isinstance(raw_attestation, Mapping), "Manifest attestation binding is missing.")
    expected_key_id = cast(Mapping[str, Any], raw_attestation).get("key_id")
    _require(contract.is_sha256(expected_key_id), "Manifest attestation key ID is invalid.")
    roots = (training_output_root, calibration_output_root, output_root)
    if attestation_key_path is None:
        trust_root = attestation.trust_root_from_environment(
            repository_root=REPOSITORY_ROOT,
            artifact_roots=roots,
            expected_key_id=cast(str, expected_key_id),
        )
    else:
        trust_root = attestation.load_trust_root(
            attestation_key_path,
            repository_root=REPOSITORY_ROOT,
            artifact_roots=roots,
            expected_key_id=cast(str, expected_key_id),
        )

    training_context = (
        calibration_matrix._load_v1_1_context(trust_root=trust_root)
        if calibration_matrix.REQUIRE_RETRY_ADMISSION
        else context
    )
    training_ledger_path, training_ledger, trainer_binding, ledger_records = (
        calibration_matrix._load_terminal_training_ledger(
            training_output_root=training_output_root,
            context=training_context,
            trust_root=trust_root,
        )
    )
    canonical_calibrator = calibration_matrix._canonical_calibration_script(
        calibration_matrix.CALIBRATION_SCRIPT
    )
    calibrator_fd, calibrator_snapshot = calibration_matrix._open_canonical_script(
        canonical_calibrator
    )
    os.close(calibrator_fd)
    calibration_ledger_path = calibration_output_root / calibration_matrix.MATRIX_SUMMARY_NAME
    _require(calibration_ledger_path.is_file(), "Terminal calibration matrix ledger is missing.")
    calibration_ledger = _load_json_nofollow(
        calibration_ledger_path, label="calibration matrix ledger"
    )
    retry_admission = calibration_matrix.load_retry_admission_for_downstream(
        output_root=calibration_output_root,
        context=context,
        training_context=training_context,
        trust_root=trust_root,
        training_matrix_summary_path=training_ledger_path,
        training_matrix_payload=training_ledger,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
    )
    calibration_records = calibration_matrix.validate_matrix_summary(
        calibration_ledger,
        output_root=calibration_output_root,
        training_output_root=training_output_root,
        calibration_script=canonical_calibrator,
        calibration_script_binding=calibrator_snapshot.public_binding,
        matrix_lock_binding=calibration_matrix._matrix_lock_binding(
            calibration_matrix._matrix_lock_path(calibration_output_root)
        ),
        context=context,
        training_context=training_context,
        trust_root=trust_root,
        training_matrix_summary_path=training_ledger_path,
        training_matrix_payload=training_ledger,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
        retry_admission=(
            None if retry_admission is None else retry_admission.public_binding
        ),
    )
    expected_calibration_cells = len(FROZEN_SCALES) * len(FROZEN_TRAINING_SEEDS)
    _require(
        calibration_ledger.get("status") == "terminal"
        and calibration_ledger.get("terminal_decision") == "GO"
        and len(calibration_records) == expected_calibration_cells,
        "Top-p launch requires a terminal all-GO calibration matrix.",
    )

    calibrations: dict[tuple[str, int], CalibrationInput] = {}
    calibration_bindings: list[dict[str, Any]] = []
    for scale in FROZEN_SCALES:
        for training_seed in FROZEN_TRAINING_SEEDS:
            path = calibration_matrix._artifact_path(calibration_output_root, scale, training_seed)
            payload = _load_json_nofollow(path, label="direct calibration artifact")
            validated = calibration_program.validate_calibration_artifact(
                payload,
                verify_bindings=True,
                trust_root=trust_root,
            )
            _require(validated == payload, "Calibration replay changed its payload.")
            _require(
                payload.get("scale") == scale
                and payload.get("training_seed") == training_seed
                and payload.get("terminal_decision") == "GO"
                and all(
                    payload.get("budget_decisions", {}).get(budget) == "GO"
                    for budget in FROZEN_BUDGETS
                ),
                f"Top-p prerequisite calibration is not all-GO: {scale}/{training_seed}.",
            )
            _require(
                payload.get("source") == context.source
                and payload.get("manifest") == context.manifest_binding,
                "Calibration source or manifest binding drifted.",
            )
            binding = _file_binding(path, payload)
            item = CalibrationInput(
                coordinate=(scale, training_seed),
                path=path,
                payload=payload,
                binding=binding,
            )
            calibrations[(scale, training_seed)] = item
            calibration_bindings.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "artifact": binding,
                }
            )
    _require(
        len(calibrations) == expected_calibration_cells,
        "Top-p calibration prerequisite inventory is incomplete.",
    )
    public_source = {
        "manifest": context.manifest_binding,
        "training_matrix": _file_binding(training_ledger_path, training_ledger),
        "calibration_matrix": _file_binding(calibration_ledger_path, calibration_ledger),
        "calibrations": calibration_bindings,
        "validated_training_cells": len(ledger_records),
        "validated_calibration_cells": len(calibration_records),
    }
    public_binding = {
        **public_source,
        "prerequisite_digest": contract.json_digest(public_source),
    }
    training_matrix.assert_environment_unchanged(context)
    return FrozenPrerequisites(
        context=context,
        trust_root=trust_root,
        calibrations=calibrations,
        public_binding=public_binding,
    )


def build_generator_command(
    *,
    generator_script: Path,
    output_root: Path,
    artifact: Path,
    calibration_input: CalibrationInput,
    coordinate: MatchCoordinate,
    device_index: int = 0,
    device_routing_identity: Mapping[str, Any],
) -> list[str]:
    canonical = _canonical_generator(generator_script)
    expected_artifact = artifact_path(output_root, coordinate)
    _require(
        _absolute(artifact) == _absolute(expected_artifact),
        "Top-p generator artifact path is not canonical.",
    )
    _require(
        calibration_input.coordinate == (coordinate.scale, coordinate.training_seed),
        "Top-p generator calibration coordinate drifted.",
    )
    _require(
        type(device_index) is int and device_index >= 0,
        "Top-p generator device index is invalid.",
    )
    routing_identity = execution_environment.validate_device_routing_identity(
        device_routing_identity
    )
    return [
        sys.executable,
        "-I",
        "-c",
        GENERATOR_FD_BOOTSTRAP,
        str(canonical),
        "--mode",
        "generate",
        "--artifact",
        str(artifact),
        "--output-root",
        str(output_root),
        "--calibration",
        str(calibration_input.path),
        "--comparator",
        coordinate.comparator,
        "--budget",
        coordinate.budget,
        "--device",
        f"cuda:{device_index}",
        "--expected-device-routing-identity-json",
        json.dumps(routing_identity, sort_keys=True, separators=(",", ":")),
    ]


def _expected_exit_code(decision: str) -> int:
    _require(decision in {"GO", "NO-GO"}, "Top-p terminal decision is invalid.")
    return 0 if decision == "GO" else 2


def _validate_generator_completion(
    result: subprocess.CompletedProcess[Any],
    *,
    artifact: Path,
    payload: Mapping[str, Any],
    execution_environment_binding: Mapping[str, Any],
) -> dict[str, Any]:
    stdout = result.stdout
    _require(isinstance(stdout, str), "Top-p generator completion evidence is missing.")
    lines = [line for line in stdout.splitlines() if line.strip()]
    _require(
        len(lines) == 1,
        "Top-p generator completion output is not one canonical JSON record.",
    )
    try:
        completion = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("Top-p generator completion evidence is invalid JSON.") from error
    _require(isinstance(completion, dict), "Top-p generator completion must be an object.")
    _require(
        set(completion) == _GENERATOR_COMPLETION_FIELDS,
        "Top-p generator completion schema drifted.",
    )
    _require(
        completion.get("artifact") == str(artifact)
        and completion.get("mode") == "generate"
        and completion.get("resumed_existing_terminal_artifact") is False,
        "Top-p generator resumed a pre-existing artifact instead of creating this cell.",
    )
    _require(
        completion.get("terminal_decision") == payload.get("terminal_decision")
        and completion.get("payload_sha256") == payload.get("payload_sha256")
        and completion.get("python") == execution_environment_binding.get("python_version")
        and completion.get("torch") == execution_environment_binding.get("torch_version"),
        "Top-p generator completion provenance drifted.",
    )
    _require(
        {
            "device_spec": completion.get("device_spec"),
            "logical_device_index": completion.get("logical_device_index"),
        }
        == execution_environment.selected_device_context(execution_environment_binding),
        "Top-p generator completion logical-device context drifted.",
    )
    return completion


def load_and_validate_artifact(
    path: Path,
    *,
    coordinate: MatchCoordinate,
    calibration_input: CalibrationInput,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _require(
        _absolute(path) == _absolute(artifact_path(path.parents[3], coordinate)),
        "Top-p artifact path does not match its frozen coordinate.",
    )
    _require(path.is_file() and not path.is_symlink(), f"Top-p artifact is missing: {path}")
    payload = _load_json_nofollow(path, label="top-p physical-match artifact")
    validated = physical_match.validate_top_p_physical_match_artifact(
        payload,
        calibration=calibration_input.payload,
        comparator=coordinate.comparator,
        budget=coordinate.budget,
        expected_scale=coordinate.scale,
        expected_training_seed=coordinate.training_seed,
        expected_calibration_seed=coordinate.calibration_seed,
        expected_global_block_budget=contract.DIRECT_GLOBAL_BLOCK_BUDGETS[coordinate.scale][
            coordinate.budget
        ],
        expected_csa_layers=contract.DIRECT_CSA_LAYERS_BY_SCALE[coordinate.scale],
        verify_bindings=True,
        trust_root=trust_root,
    )
    _require(validated == payload, "Top-p public validator changed its payload.")
    _expected_exit_code(cast(str, payload.get("terminal_decision")))
    return payload


def _cell_record(
    payload: Mapping[str, Any],
    *,
    coordinate: MatchCoordinate,
    path: Path,
    calibration_input: CalibrationInput,
    launch_nonce: str,
    command: Sequence[str],
    generator_completion: Mapping[str, Any],
) -> dict[str, Any]:
    decision = cast(str, payload.get("terminal_decision"))
    return {
        **coordinate.payload,
        "coordinate_key": coordinate.key,
        "launch_nonce": launch_nonce,
        "status": "terminal",
        "terminal_decision": decision,
        "generator_exit_code": _expected_exit_code(decision),
        "generator_completion": dict(generator_completion),
        "command": list(command),
        "calibration_input": dict(calibration_input.binding),
        "artifact": _file_binding(path, payload),
    }


def _matrix_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    prerequisites: FrozenPrerequisites,
    generator_binding: Mapping[str, Any],
    execution_environment: Mapping[str, Any],
    gpu_execution_lease_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_execution_environment = _validate_execution_environment(execution_environment)
    frozen_gpu_execution_lease = _validate_gpu_execution_lease_binding(gpu_execution_lease_binding)
    terminal = len(records) == EXPECTED_CELLS
    go_cells = sum(record.get("terminal_decision") == "GO" for record in records)
    no_go_cells = sum(record.get("terminal_decision") == "NO-GO" for record in records)
    terminal_decision: str | None = None
    if terminal:
        terminal_decision = "GO" if no_go_cells == 0 else "NO-GO"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal" if terminal else "in_progress",
        "terminal_decision": terminal_decision,
        "source": prerequisites.context.source,
        "manifest": prerequisites.context.manifest_binding,
        "attestation_contract": prerequisites.context.manifest_binding["attestation"],
        "prerequisites": dict(prerequisites.public_binding),
        "canonical_generator": dict(generator_binding),
        "execution_environment": frozen_execution_environment,
        "gpu_execution_lease": frozen_gpu_execution_lease,
        "execution_threat_boundary": EXECUTION_THREAT_BOUNDARY,
        "validation_api_semantics": VALIDATION_API_SEMANTICS,
        "matrix_lock": dict(matrix_lock_binding),
        "cell_claim_semantics": CELL_CLAIM_SEMANTICS,
        "crash_recovery_boundary": CRASH_RECOVERY_BOUNDARY,
        "outcome_policy": OUTCOME_POLICY,
        "outcome_dependent_early_stopping": False,
        "outcome_selection_performed": False,
        "coordinate_order": "scale,training_seed,budget,comparator",
        "coordinate_count": EXPECTED_CELLS,
        "coordinate_digest": coordinate_digest(),
        "completed_cells": len(records),
        "go_cells": go_cells,
        "no_go_cells": no_go_cells,
        "records": list(records),
    }
    return _attested_payload(payload, trust_root=prerequisites.trust_root)


def _validate_record_shape(record: Mapping[str, Any], coordinate: MatchCoordinate) -> None:
    _require(
        set(record) == _RECORD_FIELDS, f"Top-p matrix record schema drifted: {coordinate.key}."
    )
    for field, expected in coordinate.payload.items():
        _require(record.get(field) == expected, f"Top-p record {field} drifted: {coordinate.key}.")
    _require(record.get("coordinate_key") == coordinate.key, "Top-p coordinate key drifted.")
    _require(record.get("status") == "terminal", "Top-p matrix record is not terminal.")
    _require(contract.is_sha256(record.get("launch_nonce")), "Top-p launch nonce is invalid.")
    decision = record.get("terminal_decision")
    _require(
        decision in {"GO", "NO-GO"}
        and record.get("generator_exit_code") == _expected_exit_code(cast(str, decision)),
        "Top-p record decision/exit-code binding drifted.",
    )
    completion = record.get("generator_completion")
    _require(isinstance(completion, Mapping), "Top-p generator completion binding is missing.")
    completion = cast(Mapping[str, Any], completion)
    artifact_binding = record.get("artifact")
    _require(isinstance(artifact_binding, Mapping), "Top-p artifact binding is missing.")
    _require(
        set(completion) == _GENERATOR_COMPLETION_FIELDS
        and completion.get("mode") == "generate"
        and completion.get("resumed_existing_terminal_artifact") is False
        and completion.get("terminal_decision") == decision
        and completion.get("payload_sha256")
        == cast(Mapping[str, Any], artifact_binding).get("payload_sha256"),
        "Top-p durable create-only completion evidence drifted.",
    )
    for name in ("calibration_input", "artifact"):
        binding = record.get(name)
        _require(isinstance(binding, Mapping), f"Top-p {name} binding is missing.")
        binding = cast(Mapping[str, Any], binding)
        _require(set(binding) == _FILE_BINDING_FIELDS, f"{name} schema drifted.")
        _require(
            isinstance(binding.get("path"), str)
            and bool(binding.get("path"))
            and contract.is_sha256(binding.get("sha256"))
            and type(binding.get("bytes")) is int
            and cast(int, binding["bytes"]) > 0
            and contract.is_sha256(binding.get("payload_sha256"))
            and contract.is_sha256(binding.get("attestation_mac"))
            and isinstance(binding.get("experiment_id"), str)
            and bool(binding.get("experiment_id")),
            f"{name} binding values are invalid.",
        )


def _validate_matrix_summary_common(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    generator_script: Path,
    generator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    environment_validation: str,
    gpu_lease: GPULockLease | None,
    device_guard_lease: GPULockLease | None,
    expected_gpu_execution_lease: Mapping[str, Any] | None,
    verify_artifacts: bool = True,
) -> list[dict[str, Any]]:
    _require(
        environment_validation in {"exact", "controller-compatible", "archived"},
        "Top-p environment validation mode is invalid.",
    )
    if environment_validation == "archived":
        _require(
            gpu_lease is None and device_guard_lease is None,
            "Archived top-p validation may not borrow GPU leases.",
        )
    else:
        _require(
            gpu_lease is not None and device_guard_lease is not None,
            "Live top-p validation requires held scheduler and physical-device leases.",
        )
        cast(GPULockLease, gpu_lease).assert_held()
        cast(GPULockLease, device_guard_lease).assert_held()
    _verify_attested_payload(payload, trust_root=prerequisites.trust_root)
    if verify_artifacts:
        ledger_path = _absolute(output_root) / MATRIX_SUMMARY_NAME
        _assert_exact_json_payload_file(
            ledger_path,
            payload,
            label="Top-p matrix ledger",
        )
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Top-p matrix version drifted.")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong top-p matrix ID.")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "Top-p matrix type drifted.")
    _require(payload.get("source") == prerequisites.context.source, "Top-p source drifted.")
    _require(
        payload.get("manifest") == prerequisites.context.manifest_binding,
        "Top-p manifest binding drifted.",
    )
    _require(
        payload.get("attestation_contract")
        == prerequisites.context.manifest_binding["attestation"],
        "Top-p attestation contract drifted.",
    )
    _require(
        payload.get("prerequisites") == dict(prerequisites.public_binding),
        "Top-p prerequisite binding drifted.",
    )
    _require(
        payload.get("canonical_generator") == dict(generator_binding),
        "Top-p generator snapshot drifted.",
    )
    raw_execution_environment = payload.get("execution_environment")
    _require(
        isinstance(raw_execution_environment, Mapping),
        "Top-p execution environment binding is missing.",
    )
    frozen_execution_environment = _validate_execution_environment(
        cast(Mapping[str, Any], raw_execution_environment)
    )
    validated_gpu_execution_lease = _validate_gpu_execution_lease_binding(
        payload.get("gpu_execution_lease"),
        expected=expected_gpu_execution_lease,
    )
    _require(
        validated_gpu_execution_lease["selected_device_class"]
        == execution_environment.selected_device_class(frozen_execution_environment)
        and validated_gpu_execution_lease["selected_device_routing_identity"]
        == execution_environment.selected_device_routing_identity(frozen_execution_environment),
        "Top-p GPU execution lease is not bound to its exact selected device.",
    )
    _require(
        payload.get("execution_threat_boundary") == EXECUTION_THREAT_BOUNDARY,
        "Top-p execution threat boundary drifted.",
    )
    _require(
        payload.get("validation_api_semantics") == VALIDATION_API_SEMANTICS,
        "Top-p validation API semantics drifted.",
    )
    if environment_validation == "exact":
        _assert_execution_environment_unchanged_under_lease(
            frozen_execution_environment,
            gpu_lease=cast(GPULockLease, gpu_lease),
        )
    elif environment_validation == "controller-compatible":
        _assert_controller_compatible_execution_environment(
            frozen_execution_environment,
            gpu_lease=cast(GPULockLease, gpu_lease),
        )
    _require(
        payload.get("matrix_lock") == dict(matrix_lock_binding),
        "Top-p matrix lock binding drifted.",
    )
    if environment_validation != "archived":
        _assert_secure_regular_file(
            Path(cast(str, matrix_lock_binding["path"])),
            label="Top-p matrix lock",
        )
    _require(
        payload.get("cell_claim_semantics") == CELL_CLAIM_SEMANTICS
        and payload.get("crash_recovery_boundary") == CRASH_RECOVERY_BOUNDARY,
        "Top-p crash-recovery contract drifted.",
    )
    _require(
        payload.get("outcome_policy") == OUTCOME_POLICY
        and payload.get("outcome_dependent_early_stopping") is False
        and payload.get("outcome_selection_performed") is False,
        "Top-p outcome-independence boundary drifted.",
    )
    _require(
        payload.get("coordinate_order") == "scale,training_seed,budget,comparator"
        and payload.get("coordinate_count") == EXPECTED_CELLS
        and payload.get("coordinate_digest") == coordinate_digest(),
        "Top-p coordinate inventory drifted.",
    )
    records = payload.get("records")
    _require(isinstance(records, list), "Top-p matrix records are invalid.")
    records = cast(list[Mapping[str, Any]], records)
    _require(len(records) <= EXPECTED_CELLS, "Top-p matrix contains extra records.")
    _require(payload.get("completed_cells") == len(records), "Completed cell count drifted.")
    canonical = _canonical_generator(generator_script)
    result: list[dict[str, Any]] = []
    go_cells = 0
    no_go_cells = 0
    for record, coordinate in zip(records, coordinates()[: len(records)], strict=True):
        _require(isinstance(record, Mapping), "Top-p matrix record is invalid.")
        _validate_record_shape(record, coordinate)
        calibration_input = prerequisites.calibrations[(coordinate.scale, coordinate.training_seed)]
        path = artifact_path(output_root, coordinate)
        command = build_generator_command(
            generator_script=canonical,
            output_root=output_root,
            artifact=path,
            calibration_input=calibration_input,
            coordinate=coordinate,
            device_index=cast(int, frozen_execution_environment["current_device_index"]),
            device_routing_identity=execution_environment.selected_device_routing_identity(
                frozen_execution_environment
            ),
        )
        _require(
            record.get("calibration_input") == dict(calibration_input.binding),
            f"Top-p calibration input binding drifted: {coordinate.key}.",
        )
        _require(record.get("command") == command, f"Top-p command drifted: {coordinate.key}.")
        artifact_binding = cast(Mapping[str, Any], record["artifact"])
        _require(
            artifact_binding.get("path") == str(_absolute(path))
            and artifact_binding.get("experiment_id") == contract.DIRECT_TOP_P_MATCH_EXPERIMENT_ID,
            f"Top-p artifact path or experiment binding drifted: {coordinate.key}.",
        )
        completion = cast(Mapping[str, Any], record["generator_completion"])
        _require(
            completion.get("artifact") == str(path)
            and completion.get("python") == frozen_execution_environment["python_version"]
            and completion.get("torch") == frozen_execution_environment["torch_version"]
            and {
                "device_spec": completion.get("device_spec"),
                "logical_device_index": completion.get("logical_device_index"),
            }
            == execution_environment.selected_device_context(frozen_execution_environment),
            f"Top-p durable generator provenance drifted: {coordinate.key}.",
        )
        if verify_artifacts:
            loaded = load_and_validate_artifact(
                path,
                coordinate=coordinate,
                calibration_input=calibration_input,
                trust_root=prerequisites.trust_root,
            )
            expected = _cell_record(
                loaded,
                coordinate=coordinate,
                path=path,
                calibration_input=calibration_input,
                launch_nonce=cast(str, record["launch_nonce"]),
                command=command,
                generator_completion=completion,
            )
            _require(record == expected, f"Top-p matrix record drifted: {coordinate.key}.")
        decision = record.get("terminal_decision")
        go_cells += decision == "GO"
        no_go_cells += decision == "NO-GO"
        result.append(dict(record))
    terminal = len(records) == EXPECTED_CELLS
    expected_terminal_decision: str | None = None
    if terminal:
        expected_terminal_decision = "GO" if no_go_cells == 0 else "NO-GO"
    _require(
        payload.get("status") == ("terminal" if terminal else "in_progress")
        and payload.get("terminal_decision") == expected_terminal_decision,
        "Top-p matrix terminal state drifted.",
    )
    _require(
        payload.get("go_cells") == go_cells and payload.get("no_go_cells") == no_go_cells,
        "Top-p matrix decision counts drifted.",
    )
    _preflight_output_tree(
        output_root=output_root,
        matrix_summary=output_root / MATRIX_SUMMARY_NAME,
        completed_cells=len(records),
    )
    return result


def _validate_matrix_summary_under_gpu_lease(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    generator_script: Path,
    generator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    gpu_lease: GPULockLease,
    device_guard_lease: GPULockLease,
    verify_artifacts: bool = True,
) -> list[dict[str, Any]]:
    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    current_execution_environment = _capture_execution_environment_under_lease(gpu_lease)
    _assert_device_guard_matches_environment(
        device_guard_lease,
        current_execution_environment,
    )
    binding = _gpu_execution_lease_binding(
        gpu_lease,
        device_guard_lease=device_guard_lease,
        current_execution_environment=current_execution_environment,
    )
    result = _validate_matrix_summary_common(
        payload,
        output_root=output_root,
        prerequisites=prerequisites,
        generator_script=generator_script,
        generator_binding=generator_binding,
        matrix_lock_binding=matrix_lock_binding,
        environment_validation="exact",
        gpu_lease=gpu_lease,
        device_guard_lease=device_guard_lease,
        expected_gpu_execution_lease=binding,
        verify_artifacts=verify_artifacts,
    )
    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    return result


def validate_matrix_summary(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    generator_script: Path,
    generator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    verify_artifacts: bool = True,
    gpu_lock_path: Path | None = None,
    gpu_lease: GPULockLease | None = None,
    device_guard_lease: GPULockLease | None = None,
) -> list[dict[str, Any]]:
    """Exact same-host replay, with either one borrowed lease or one owned lease."""

    owned_lease: GPULockLease | None = None
    owned_device_guard: GPULockLease | None = None
    lease = gpu_lease
    if lease is None:
        requested_path = (
            contract.DIRECT_GPU_SCHEDULER_LOCK_PATH
            if gpu_lock_path is None
            else gpu_lock_path
        )
        _require(
            not _paths_overlap(
                _absolute(requested_path).resolve(strict=False),
                _absolute(calibration_matrix.SUPERSEDED_OUTPUT_ROOT).resolve(strict=False),
            ),
            "Top-p validation GPU lock may not overlap the immutable calibration quarantine.",
        )
        owned_lease = acquire_gpu_lock(
            "p2-direct-top-p-physical-matrix-public-validation",
            path=requested_path,
        )
        lease = owned_lease
    else:
        lease.assert_held()
        if gpu_lock_path is not None:
            requested_path = _exact_resolved_path(
                gpu_lock_path,
                label="Borrowed top-p GPU execution lease",
            )
            _require(
                requested_path == lease.path,
                "Borrowed top-p GPU lease does not match gpu_lock_path.",
            )

    _require(
        not _paths_overlap(
            _absolute(lease.path).resolve(strict=False),
            _absolute(calibration_matrix.SUPERSEDED_OUTPUT_ROOT).resolve(strict=False),
        ),
        "Top-p GPU lease may not overlap the immutable calibration quarantine.",
    )

    try:
        current_execution_environment = _capture_execution_environment_under_lease(lease)
        routing_identity = execution_environment.selected_device_routing_identity(
            current_execution_environment
        )
        guard = device_guard_lease
        if guard is None:
            owned_device_guard = acquire_device_guard(
                "p2-direct-top-p-physical-matrix-public-validation",
                routing_identity,
            )
            guard = owned_device_guard
        _assert_device_guard_matches_environment(guard, current_execution_environment)
        return _validate_matrix_summary_under_gpu_lease(
            payload,
            output_root=output_root,
            prerequisites=prerequisites,
            generator_script=generator_script,
            generator_binding=generator_binding,
            matrix_lock_binding=matrix_lock_binding,
            gpu_lease=lease,
            device_guard_lease=guard,
            verify_artifacts=verify_artifacts,
        )
    finally:
        if owned_device_guard is not None:
            owned_device_guard.close()
        if owned_lease is not None:
            owned_lease.close()


def validate_matrix_summary_controller_compatible(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    generator_script: Path,
    generator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    gpu_lease: GPULockLease,
    device_guard_lease: GPULockLease,
    verify_artifacts: bool = True,
) -> list[dict[str, Any]]:
    """Replay under a caller-held lease on a different GPU of the exact frozen hardware class."""

    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    current_execution_environment = _capture_execution_environment_under_lease(gpu_lease)
    _assert_device_guard_matches_environment(
        device_guard_lease,
        current_execution_environment,
    )
    result = _validate_matrix_summary_common(
        payload,
        output_root=output_root,
        prerequisites=prerequisites,
        generator_script=generator_script,
        generator_binding=generator_binding,
        matrix_lock_binding=matrix_lock_binding,
        environment_validation="controller-compatible",
        gpu_lease=gpu_lease,
        device_guard_lease=device_guard_lease,
        expected_gpu_execution_lease=None,
        verify_artifacts=verify_artifacts,
    )
    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    return result


def validate_matrix_summary_archived(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    generator_script: Path,
    generator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    verify_artifacts: bool = True,
) -> list[dict[str, Any]]:
    """Stored-only HMAC/schema/artifact replay with no live CUDA or GPU-lease operation."""

    return _validate_matrix_summary_common(
        payload,
        output_root=output_root,
        prerequisites=prerequisites,
        generator_script=generator_script,
        generator_binding=generator_binding,
        matrix_lock_binding=matrix_lock_binding,
        environment_validation="archived",
        gpu_lease=None,
        device_guard_lease=None,
        expected_gpu_execution_lease=None,
        verify_artifacts=verify_artifacts,
    )


def _preflight_output_tree(
    *, output_root: Path, matrix_summary: Path, completed_cells: int
) -> None:
    root = _absolute(output_root)
    summary = _absolute(matrix_summary)
    if not root.exists():
        _require(completed_cells == 0, "Completed top-p prefix lost its output root.")
        return
    _require(root.is_dir() and not root.is_symlink(), "Top-p output root is unsafe.")
    allowed: set[Path] = {summary}
    if _path_lexists(summary):
        _assert_secure_regular_file(summary, label="Top-p matrix ledger")
    for index, coordinate in enumerate(coordinates()):
        path = artifact_path(root, coordinate)
        output_dir = path.parent
        cursor = output_dir
        while cursor != root:
            allowed.add(cursor)
            cursor = cursor.parent
        claim_path = output_dir / CELL_CLAIM_NAME
        if index < completed_cells:
            allowed.add(path)
            _require(_path_lexists(path), f"Completed top-p artifact is missing: {coordinate.key}.")
            _assert_secure_regular_file(
                path,
                label=f"Completed top-p artifact {coordinate.key}",
            )
            _require(
                not _path_lexists(claim_path),
                f"Completed top-p cell retained a claim: {coordinate.key}.",
            )
        else:
            _require(
                not _path_lexists(path) and not _path_lexists(claim_path),
                f"Orphan top-p artifact or claim blocks resume: {coordinate.key}.",
            )
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"Top-p output tree contains a symlink: {item}")
        _require(
            _absolute(item) in allowed,
            f"Top-p output tree contains an unregistered orphan: {item}",
        )


def _assert_empty_cell(output_root: Path, coordinate: MatchCoordinate) -> None:
    path = artifact_path(output_root, coordinate)
    _require(
        not _path_lexists(path),
        f"Refusing pre-existing top-p artifact: {coordinate.key}.",
    )
    _require(
        not _path_lexists(path.parent / CELL_CLAIM_NAME),
        f"Refusing orphan top-p cell claim: {coordinate.key}.",
    )


def _run_matrix_under_gpu_lease(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = TRAINING_OUTPUT_ROOT,
    calibration_output_root: Path = CALIBRATION_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path | None = None,
    generator_script: Path = GENERATOR_SCRIPT,
    attestation_key_path: Path | None = None,
    max_new_cells: int | None = None,
    gpu_lease: GPULockLease,
    device_guard_lease: GPULockLease,
    current_execution_environment: Mapping[str, Any],
) -> dict[str, Any]:
    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    _assert_device_guard_matches_environment(
        device_guard_lease,
        current_execution_environment,
    )
    gpu_execution_lease_binding = _gpu_execution_lease_binding(
        gpu_lease,
        device_guard_lease=device_guard_lease,
        current_execution_environment=current_execution_environment,
    )
    if max_new_cells is not None:
        _require(
            type(max_new_cells) is int and max_new_cells > 0,
            "max-new-cells must be a positive integer.",
        )
    key_path_for_layout = attestation_key_path
    if key_path_for_layout is None and os.environ.get(attestation.KEY_PATH_ENV):
        key_path_for_layout = Path(os.environ[attestation.KEY_PATH_ENV])
    requested_summary = (
        output_root / MATRIX_SUMMARY_NAME if matrix_summary is None else matrix_summary
    )
    layout = _validate_matrix_layout(
        output_root=output_root,
        matrix_summary=requested_summary,
        training_output_root=training_output_root,
        calibration_output_root=calibration_output_root,
        attestation_key_path=key_path_for_layout,
    )
    canonical = _canonical_generator(generator_script)
    lock_binding = _matrix_lock_binding(layout.lock_path)
    gpu_lease.assert_held()
    with _exclusive_matrix_lock(
        layout.lock_path, matrix_summary=layout.matrix_summary
    ) as assert_matrix_lock_held:
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
        prerequisites = load_and_validate_prerequisites(
            manifest_path=manifest_path,
            training_output_root=layout.training_output_root,
            calibration_output_root=layout.calibration_output_root,
            output_root=layout.output_root,
            attestation_key_path=attestation_key_path,
        )
        _assert_implementation_binding(prerequisites.context)
        opened_generator, generator_snapshot = _open_generator(canonical)
        opened_generator.close()
        generator_binding = generator_snapshot.public_binding

        completed: list[dict[str, Any]] = []
        if _path_lexists(layout.matrix_summary):
            existing_ledger = _load_json_nofollow(
                layout.matrix_summary,
                label="top-p matrix ledger",
            )
            completed = _validate_matrix_summary_under_gpu_lease(
                existing_ledger,
                output_root=layout.output_root,
                prerequisites=prerequisites,
                generator_script=canonical,
                generator_binding=generator_binding,
                matrix_lock_binding=lock_binding,
                gpu_lease=gpu_lease,
                device_guard_lease=device_guard_lease,
                verify_artifacts=True,
            )
            raw_execution_environment = existing_ledger.get("execution_environment")
            _require(
                isinstance(raw_execution_environment, Mapping),
                "Top-p resume execution environment is missing.",
            )
            frozen_execution_environment = _validate_execution_environment(
                cast(Mapping[str, Any], raw_execution_environment)
            )
        else:
            frozen_execution_environment = _validate_execution_environment(
                current_execution_environment
            )
        _preflight_output_tree(
            output_root=layout.output_root,
            matrix_summary=layout.matrix_summary,
            completed_cells=len(completed),
        )
        training_matrix.assert_environment_unchanged(prerequisites.context)
        _assert_execution_environment_unchanged_under_lease(
            frozen_execution_environment,
            gpu_lease=gpu_lease,
        )
        verification, _snapshot = _open_generator(canonical, expected=generator_snapshot)
        verification.close()
        if not _path_lexists(layout.matrix_summary):
            assert_matrix_lock_held()
            _assert_execution_environment_unchanged_under_lease(
                frozen_execution_environment,
                gpu_lease=gpu_lease,
            )
            gpu_lease.assert_held()
            _atomic_write_json(
                layout.matrix_summary,
                _matrix_payload(
                    completed,
                    prerequisites=prerequisites,
                    generator_binding=generator_binding,
                    execution_environment=frozen_execution_environment,
                    gpu_execution_lease_binding=gpu_execution_lease_binding,
                    matrix_lock_binding=lock_binding,
                ),
            )
            gpu_lease.assert_held()
            assert_matrix_lock_held()

        new_cells = 0
        for index, coordinate in enumerate(coordinates()):
            if index < len(completed):
                continue
            if max_new_cells is not None and new_cells >= max_new_cells:
                break
            training_matrix.assert_environment_unchanged(prerequisites.context)
            _assert_execution_environment_unchanged_under_lease(
                frozen_execution_environment,
                gpu_lease=gpu_lease,
            )
            _assert_empty_cell(layout.output_root, coordinate)
            calibration_input = prerequisites.calibrations[
                (coordinate.scale, coordinate.training_seed)
            ]
            path = artifact_path(layout.output_root, coordinate)
            launch_nonce = secrets.token_hex(32)
            command = build_generator_command(
                generator_script=canonical,
                output_root=layout.output_root,
                artifact=path,
                calibration_input=calibration_input,
                coordinate=coordinate,
                device_index=cast(int, frozen_execution_environment["current_device_index"]),
                device_routing_identity=(
                    execution_environment.selected_device_routing_identity(
                        frozen_execution_environment
                    )
                ),
            )
            with _exclusive_cell_claim(
                path.parent,
                coordinate=coordinate,
                launch_nonce=launch_nonce,
            ):
                _require(
                    not _path_lexists(path),
                    f"Refusing top-p artifact created before child launch: {coordinate.key}.",
                )
                gpu_lease.assert_held()
                device_guard_lease.assert_held()
                result = _run_generator_from_snapshot(
                    command,
                    canonical=canonical,
                    expected=generator_snapshot,
                    trust_root=prerequisites.trust_root,
                    gpu_lease=gpu_lease,
                    device_guard_lease=device_guard_lease,
                )
                gpu_lease.assert_held()
                device_guard_lease.assert_held()
                training_matrix.assert_environment_unchanged(prerequisites.context)
                _assert_execution_environment_unchanged_under_lease(
                    frozen_execution_environment,
                    gpu_lease=gpu_lease,
                )
                if not _path_lexists(path):
                    if result.returncode not in {0, 2}:
                        raise subprocess.CalledProcessError(result.returncode, command)
                    raise ValueError(
                        f"Top-p generator did not publish its terminal artifact: {coordinate.key}."
                    )
                before = _file_binding(path, _load_json_nofollow(path, label="top-p artifact"))
                payload = load_and_validate_artifact(
                    path,
                    coordinate=coordinate,
                    calibration_input=calibration_input,
                    trust_root=prerequisites.trust_root,
                )
                after = _file_binding(path, payload)
                _require(before == after, "Top-p artifact changed during public validation.")
                _require(
                    result.returncode
                    == _expected_exit_code(cast(str, payload["terminal_decision"])),
                    "Top-p generator exit code does not match its attested decision.",
                )
                generator_completion = _validate_generator_completion(
                    result,
                    artifact=path,
                    payload=payload,
                    execution_environment_binding=frozen_execution_environment,
                )
                record = _cell_record(
                    payload,
                    coordinate=coordinate,
                    path=path,
                    calibration_input=calibration_input,
                    launch_nonce=launch_nonce,
                    command=command,
                    generator_completion=generator_completion,
                )
            completed.append(record)
            _preflight_output_tree(
                output_root=layout.output_root,
                matrix_summary=layout.matrix_summary,
                completed_cells=len(completed),
            )
            assert_matrix_lock_held()
            _assert_execution_environment_unchanged_under_lease(
                frozen_execution_environment,
                gpu_lease=gpu_lease,
            )
            gpu_lease.assert_held()
            _atomic_write_json(
                layout.matrix_summary,
                _matrix_payload(
                    completed,
                    prerequisites=prerequisites,
                    generator_binding=generator_binding,
                    execution_environment=frozen_execution_environment,
                    gpu_execution_lease_binding=gpu_execution_lease_binding,
                    matrix_lock_binding=lock_binding,
                ),
            )
            gpu_lease.assert_held()
            assert_matrix_lock_held()
            new_cells += 1

        result_payload = _matrix_payload(
            completed,
            prerequisites=prerequisites,
            generator_binding=generator_binding,
            execution_environment=frozen_execution_environment,
            gpu_execution_lease_binding=gpu_execution_lease_binding,
            matrix_lock_binding=lock_binding,
        )
        assert_matrix_lock_held()
        _assert_execution_environment_unchanged_under_lease(
            frozen_execution_environment,
            gpu_lease=gpu_lease,
        )
        gpu_lease.assert_held()
        _atomic_write_json(layout.matrix_summary, result_payload)
        gpu_lease.assert_held()
        assert_matrix_lock_held()
        on_disk = _load_json_nofollow(
            layout.matrix_summary,
            label="published top-p matrix ledger",
        )
        _require(on_disk == result_payload, "Published top-p matrix ledger bytes drifted.")
        _validate_matrix_summary_under_gpu_lease(
            on_disk,
            output_root=layout.output_root,
            prerequisites=prerequisites,
            generator_script=canonical,
            generator_binding=generator_binding,
            matrix_lock_binding=lock_binding,
            gpu_lease=gpu_lease,
            device_guard_lease=device_guard_lease,
            verify_artifacts=True,
        )
        training_matrix.assert_environment_unchanged(prerequisites.context)
        _assert_execution_environment_unchanged_under_lease(
            frozen_execution_environment,
            gpu_lease=gpu_lease,
        )
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
        return result_payload


def run_matrix(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = TRAINING_OUTPUT_ROOT,
    calibration_output_root: Path = CALIBRATION_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path | None = None,
    generator_script: Path = GENERATOR_SCRIPT,
    attestation_key_path: Path | None = None,
    max_new_cells: int | None = None,
    gpu_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Run the CUDA matrix while one process owns the project-wide GPU lease."""

    requested_gpu_lock = (
        contract.DIRECT_GPU_SCHEDULER_LOCK_PATH
        if gpu_lock_path is None
        else gpu_lock_path
    )
    _require(
        not _paths_overlap(
            _absolute(requested_gpu_lock).resolve(strict=False),
            _absolute(calibration_matrix.SUPERSEDED_OUTPUT_ROOT).resolve(strict=False),
        ),
        "Top-p GPU lock may not overlap the immutable calibration quarantine.",
    )
    gpu_lease = acquire_gpu_lock(
        "p2-direct-top-p-physical-matrix",
        path=requested_gpu_lock,
    )
    device_guard_lease: GPULockLease | None = None
    try:
        gpu_lease.assert_held()
        current_execution_environment = _capture_execution_environment_under_lease(gpu_lease)
        routing_identity = execution_environment.selected_device_routing_identity(
            current_execution_environment
        )
        device_guard_lease = acquire_device_guard(
            "p2-direct-top-p-physical-matrix",
            routing_identity,
        )
        device_guard_lease.assert_held()
        return _run_matrix_under_gpu_lease(
            manifest_path=manifest_path,
            training_output_root=training_output_root,
            calibration_output_root=calibration_output_root,
            output_root=output_root,
            matrix_summary=matrix_summary,
            generator_script=generator_script,
            attestation_key_path=attestation_key_path,
            max_new_cells=max_new_cells,
            gpu_lease=gpu_lease,
            device_guard_lease=device_guard_lease,
            current_execution_environment=current_execution_environment,
        )
    finally:
        if device_guard_lease is not None:
            device_guard_lease.close()
        gpu_lease.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume-safe frozen 40-cell direct top-p physical-match matrix.",
        epilog=f"{EXECUTION_THREAT_BOUNDARY}\n\n{VALIDATION_API_SEMANTICS}",
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--training-output-root", type=Path, default=TRAINING_OUTPUT_ROOT)
    parser.add_argument("--calibration-output-root", type=Path, default=CALIBRATION_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--matrix-summary", type=Path)
    parser.add_argument(
        "--gpu-lock-path",
        type=Path,
        help="Persistent device-scoped GPU lease path; all same-device work must share it.",
    )
    parser.add_argument(
        "--max-new-cells",
        type=int,
        help=(
            "Operational prefix boundary only; resume requires the same controlled single-owner "
            "host, exact execution environment, clean implementation tree, and authenticated "
            "committed prefix."
        ),
    )
    args = parser.parse_args()
    result = run_matrix(
        manifest_path=args.manifest,
        training_output_root=args.training_output_root,
        calibration_output_root=args.calibration_output_root,
        output_root=args.output_root,
        matrix_summary=args.matrix_summary,
        generator_script=GENERATOR_SCRIPT,
        max_new_cells=args.max_new_cells,
        gpu_lock_path=args.gpu_lock_path,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "terminal_decision": result["terminal_decision"],
                "completed_cells": result["completed_cells"],
                "go_cells": result["go_cells"],
                "no_go_cells": result["no_go_cells"],
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 2 if result["status"] == "terminal" and result["terminal_decision"] == "NO-GO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
