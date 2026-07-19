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

import calibrate_p2_direct_soft_lag as calibration
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import run_p2_direct_training_matrix as training_matrix

EXPERIMENT_ID = "p2-post-rank-direct-soft-lag-calibration-matrix-v1"
ARTIFACT_TYPE = "direct-soft-lag-calibration-matrix"
SCHEMA_VERSION = 2

FROZEN_SCALES = ("s55", "s151")
FROZEN_TRAINING_SEEDS = (6071406, 6071407, 6071408, 6071409, 6071410)
FROZEN_CALIBRATION_SEEDS = (7071406, 7071407, 7071408, 7071409, 7071410)
EXPECTED_CELLS = len(FROZEN_SCALES) * len(FROZEN_TRAINING_SEEDS)

TRAINING_OUTPUT_ROOT = training_matrix.OUTPUT_ROOT
OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/calibration")
MATRIX_SUMMARY_NAME = "calibration-matrix.summary.json"
MATRIX_SUMMARY = OUTPUT_ROOT / MATRIX_SUMMARY_NAME
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_SCRIPT = Path(__file__).resolve().with_name("calibrate_p2_direct_soft_lag.py")
CALIBRATION_IMPLEMENTATION_PATH = CALIBRATION_SCRIPT.relative_to(REPOSITORY_ROOT).as_posix()
DEVICE = "cuda"
DTYPE = "bfloat16"
CALIBRATOR_PUBLICATION_SEMANTICS = "exclusive-atomic-mac-attested-full-external-binding"
CALIBRATOR_EXECUTION_SEMANTICS = "canonical-manifest-bound-sealed-memfd-v1"
CALIBRATOR_FD_ENV = "ADAPTIVE_V4_CANONICAL_CALIBRATOR_FD"
MATRIX_LOCK_SUFFIX = "p2-direct-calibration-matrix.lock"
MATRIX_LOCK_SEMANTICS = "persistent-sibling-flock-exclusive-process-owner-v1"
MATRIX_ATTESTATION_PURPOSE = "p2-direct-soft-lag-calibration-matrix-v1"
CRASH_RECOVERY_BOUNDARY = (
    "fail-closed: an exclusively published calibration artifact that is not present in the "
    "digest-bound completed matrix prefix is never auto-promoted; recovery requires an "
    "independently MAC-attested per-cell journal or explicit operator quarantine and rerun"
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
def _exclusive_matrix_lock(lock_path: Path, *, matrix_summary: Path) -> Iterator[dict[str, Any]]:
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
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        acquired = True
        current = os.stat(lock_path, follow_symlinks=False)
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Calibration matrix lock path was replaced before ownership.",
        )
        _write_lock_metadata(file_descriptor, owner)
        try:
            yield dict(owner)
        except BaseException:
            outcome = "released-after-error"
            raise
        finally:
            released = {
                **owner,
                "state": outcome,
                "released_time_ns": time.time_ns(),
            }
            try:
                _write_lock_metadata(file_descriptor, released)
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
) -> subprocess.CompletedProcess[Any]:
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
            pass_fds=(sealed_descriptor, key_descriptor),
            env=environment,
        )
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
    payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    digest_bound = _digest_bound_payload(payload)
    digest_bound["attestation"] = attestation.attest_payload(
        digest_bound,
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
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
    summary = training_matrix.load_validated_training_summary(
        summary_path,
        output_root=training_output_root,
        train_script=training_matrix.TRAIN_SCRIPT,
        context=context,
        scale=scale,
        seed=training_seed,
        trust_root=trust_root,
        launch_nonce=cast(str, ledger_record["launch_nonce"]),
        trainer_sha256=cast(str, trainer_binding["sha256"]),
    )
    checkpoint = summary.get("checkpoint")
    _require(isinstance(checkpoint, dict), "Validated training checkpoint binding is missing.")
    _require(
        ledger_record.get("checkpoint") == checkpoint,
        "Terminal training ledger checkpoint binding drifted.",
    )
    return summary_path, summary, cast(dict[str, Any], checkpoint)


def build_calibration_command(
    *,
    calibration_script: Path,
    artifact_path: Path,
    manifest_path: Path,
    training_summary_path: Path,
    training_matrix_summary_path: Path,
    checkpoint_path: Path,
    scale: str,
    training_seed: int,
) -> list[str]:
    _require(scale in FROZEN_SCALES, "Calibration command scale is not frozen.")
    _require(training_seed in FROZEN_TRAINING_SEEDS, "Calibration command seed is not frozen.")
    canonical = _canonical_calibration_script(calibration_script)
    return [
        sys.executable,
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
        "--training-summary",
        str(training_summary_path.resolve()),
        "--training-matrix-summary",
        str(training_matrix_summary_path.resolve()),
        "--device",
        DEVICE,
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
) -> dict[str, Any]:
    return {
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


def _expected_checkpoint_binding(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    path = checkpoint.get("path")
    if not isinstance(path, str):
        raise ValueError("Validated checkpoint path is invalid.")
    return {
        "path": str(Path(path).resolve()),
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
    trust_root: attestation.TrustRoot,
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
        "device": DEVICE,
        "dtype": DTYPE,
        "calibration_script": dict(calibration_script_binding),
        "calibrator_publication_semantics": CALIBRATOR_PUBLICATION_SEMANTICS,
        "calibrator_execution_semantics": CALIBRATOR_EXECUTION_SEMANTICS,
        "matrix_lock": dict(matrix_lock_binding),
        "crash_recovery_boundary": CRASH_RECOVERY_BOUNDARY,
        "expected_cells": len(coordinates),
        "completed_cells": len(cells),
        "cell_decisions": decisions,
        "cells": list(cells),
        "quality_evaluation_started": QUALITY_EVALUATION_STARTED,
        "quality_gate": QUALITY_GATE,
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
    context: training_matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
    training_matrix_summary_path: Path,
    training_matrix_payload: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    ledger_records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
    _require(payload.get("device") == DEVICE, "Calibration matrix is not CUDA-bound.")
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

    validated: list[dict[str, Any]] = []
    decisions: dict[str, str] = {}
    for stored_record, coordinate in zip(cells, coordinates[: len(cells)], strict=True):
        scale, training_seed, calibration_seed, evaluation_seed = coordinate
        summary_path, summary, checkpoint = _validate_training_input(
            training_output_root=training_output_root,
            scale=scale,
            training_seed=training_seed,
            context=context,
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
            training_summary_path=summary_path,
            training_matrix_summary_path=training_matrix_summary_path,
            checkpoint_path=checkpoint_path,
            scale=scale,
            training_seed=training_seed,
        )
        artifact = load_and_validate_calibration_artifact(
            artifact_path,
            scale=scale,
            training_seed=training_seed,
            calibration_seed=calibration_seed,
            evaluation_seed=evaluation_seed,
            context=context,
            summary_path=summary_path,
            summary=summary,
            checkpoint=checkpoint,
            command=command,
            trust_root=trust_root,
            training_matrix_summary_path=training_matrix_summary_path,
            training_matrix_payload=training_matrix_payload,
            ledger_record=ledger_records[(scale, training_seed)],
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
    return validated


def _assert_empty_cell_output(output_dir: Path) -> None:
    if output_dir.exists():
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
        training_matrix.assert_environment_unchanged(context)
        return ledger_path, payload, trainer_binding, ledger_records
    finally:
        opened_ledger.close()
        opened_trainer.close()


def _preflight_output_tree(
    *,
    output_root: Path,
    matrix_summary: Path,
    completed_cells: int,
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
    attestation_key_path: Path | None,
) -> dict[str, Any]:
    lock_path = str(matrix_lock_binding.get("path"))
    with _ACTIVE_MATRIX_LOCKS_GUARD:
        lock_owned = lock_path in _ACTIVE_MATRIX_LOCKS
    _require(lock_owned, "Calibration matrix execution requires the exclusive process lock.")
    canonical_script = _canonical_calibration_script(calibration_script)
    context = training_matrix.establish_frozen_context(manifest_path)
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
    (
        training_matrix_summary_path,
        training_matrix_payload,
        trainer_binding,
        ledger_records,
    ) = _load_terminal_training_ledger(
        training_output_root=training_output_root,
        context=context,
        trust_root=trust_root,
    )

    completed: list[dict[str, Any]] = []
    if matrix_summary.exists():
        completed = validate_matrix_summary(
            _load_json(matrix_summary, label="calibration matrix"),
            output_root=output_root,
            training_output_root=training_output_root,
            calibration_script=canonical_script,
            calibration_script_binding=script_binding,
            matrix_lock_binding=matrix_lock_binding,
            context=context,
            trust_root=trust_root,
            training_matrix_summary_path=training_matrix_summary_path,
            training_matrix_payload=training_matrix_payload,
            trainer_binding=trainer_binding,
            ledger_records=ledger_records,
        )

    _preflight_output_tree(
        output_root=output_root,
        matrix_summary=matrix_summary,
        completed_cells=len(completed),
    )
    training_inputs = _preflight_training_inputs(
        training_output_root=training_output_root,
        context=context,
        trust_root=trust_root,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
    )
    training_matrix.assert_environment_unchanged(context)
    verification_fd, _ = _open_canonical_script(canonical_script, expected=script_snapshot)
    os.close(verification_fd)

    if not matrix_summary.exists():
        _atomic_write_json(
            matrix_summary,
            _matrix_payload(
                completed,
                context=context,
                calibration_script_binding=script_binding,
                matrix_lock_binding=matrix_lock_binding,
                trust_root=trust_root,
            ),
        )

    for index, (scale, training_seed, calibration_seed, evaluation_seed) in enumerate(
        _coordinates()
    ):
        training_matrix.assert_environment_unchanged(context)
        summary_path, summary, checkpoint = training_inputs[(scale, training_seed)]
        artifact_path = _artifact_path(output_root, scale, training_seed)
        checkpoint_path = Path(cast(str, checkpoint["path"]))
        command = build_calibration_command(
            calibration_script=canonical_script,
            artifact_path=artifact_path,
            manifest_path=context.manifest_path,
            training_summary_path=summary_path,
            training_matrix_summary_path=training_matrix_summary_path,
            checkpoint_path=checkpoint_path,
            scale=scale,
            training_seed=training_seed,
        )

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
                context=context,
                trust_root=trust_root,
                trainer_binding=trainer_binding,
                ledger_record=ledger_records[(scale, training_seed)],
            )
            checkpoint_path = Path(cast(str, checkpoint["path"]))
            command = build_calibration_command(
                calibration_script=canonical_script,
                artifact_path=artifact_path,
                manifest_path=context.manifest_path,
                training_summary_path=summary_path,
                training_matrix_summary_path=training_matrix_summary_path,
                checkpoint_path=checkpoint_path,
                scale=scale,
                training_seed=training_seed,
            )
            result = _run_calibrator_from_stable_script(
                command,
                canonical=canonical_script,
                expected=script_snapshot,
                trust_root=trust_root,
            )
            training_matrix.assert_environment_unchanged(context)
            if not artifact_path.is_file():
                if result.returncode not in {0, 2}:
                    raise subprocess.CalledProcessError(result.returncode, command)
                raise ValueError(
                    f"Calibrator did not publish its exclusive artifact: {scale}/{training_seed}."
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
                _sha256(artifact_path) == published_sha256
                and artifact_path.stat().st_size == published_bytes,
                "Exclusive calibration artifact changed during validation.",
            )
            _require(
                result.returncode == _expected_exit_code(cast(str, artifact["terminal_decision"])),
                "Calibrator exit code does not match its MAC-attested terminal decision.",
            )

        record = _cell_record(artifact, artifact_path=artifact_path, command=command)
        if index < len(completed):
            _require(
                completed[index] == record,
                f"Previously completed calibration cell drifted: {scale}/{training_seed}.",
            )
        else:
            completed.append(record)
            _atomic_write_json(
                matrix_summary,
                _matrix_payload(
                    completed,
                    context=context,
                    calibration_script_binding=script_binding,
                    matrix_lock_binding=matrix_lock_binding,
                    trust_root=trust_root,
                ),
            )

    training_matrix.assert_environment_unchanged(context)
    terminal = _matrix_payload(
        completed,
        context=context,
        calibration_script_binding=script_binding,
        matrix_lock_binding=matrix_lock_binding,
        trust_root=trust_root,
    )
    _atomic_write_json(matrix_summary, terminal)
    validate_matrix_summary(
        terminal,
        output_root=output_root,
        training_output_root=training_output_root,
        calibration_script=canonical_script,
        calibration_script_binding=script_binding,
        matrix_lock_binding=matrix_lock_binding,
        context=context,
        trust_root=trust_root,
        training_matrix_summary_path=training_matrix_summary_path,
        training_matrix_payload=training_matrix_payload,
        trainer_binding=trainer_binding,
        ledger_records=ledger_records,
    )
    return terminal


def run_matrix(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = TRAINING_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path | None = None,
    calibration_script: Path = CALIBRATION_SCRIPT,
    attestation_key_path: Path | None = None,
) -> dict[str, Any]:
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
    with _exclusive_matrix_lock(layout.lock_path, matrix_summary=layout.matrix_summary):
        return _run_matrix_locked(
            manifest_path=manifest_path,
            training_output_root=layout.training_output_root,
            output_root=layout.output_root,
            matrix_summary=layout.matrix_summary,
            calibration_script=canonical_script,
            matrix_lock_binding=lock_binding,
            attestation_key_path=attestation_key_path,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume-safe launcher for the frozen post-rank direct SoftLag calibration grid."
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--training-output-root", type=Path, default=TRAINING_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--matrix-summary", type=Path)
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
