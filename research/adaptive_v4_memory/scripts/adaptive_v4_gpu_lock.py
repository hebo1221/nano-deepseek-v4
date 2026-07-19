from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

DEFAULT_LOCK_PATH = Path("/tmp/nano-deepseek-v4-adaptive-v4-memory-gpu.lock")
DEVICE_GUARD_PREFIX = "nano-deepseek-v4-adaptive-v4-memory-device-guard-"
DEVICE_GUARD_SEMANTICS = "selected-physical-device-identity-derived-exclusive-flock-v1"
SAFE_LOCK_MODE = 0o600


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_device_guard_path(routing_identity: Mapping[str, Any]) -> Path:
    """Derive one host-local kernel-lock path for one selected physical GPU."""

    _require(
        set(routing_identity) == {"identity_type", "identity"},
        "Adaptive V4 GPU routing identity schema drifted.",
    )
    identity_type = routing_identity.get("identity_type")
    identity = routing_identity.get("identity")
    _require(
        identity_type in {"uuid", "pci_bus_id"}
        and isinstance(identity, str)
        and bool(identity)
        and identity == identity.strip()
        and "\x00" not in identity
        and "\n" not in identity
        and "\r" not in identity
        and len(identity) <= 512,
        "Adaptive V4 GPU routing identity is invalid.",
    )
    digest = hashlib.sha256(f"{identity_type}\0{identity}".encode()).hexdigest()
    return DEFAULT_LOCK_PATH.parent / f"{DEVICE_GUARD_PREFIX}{digest}.lock"


@dataclass
class GPULockLease:
    """One persistent-inode GPU lease whose path identity is checked on release."""

    path: Path
    file_descriptor: int
    device: int
    inode: int
    ctime_ns: int
    mtime_ns: int
    holder: bytes
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def fileno(self) -> int:
        _require(not self._closed, "Adaptive V4 GPU lease is already closed.")
        return self.file_descriptor

    def assert_held(self) -> None:
        _require(not self._closed, "Adaptive V4 GPU lease is already closed.")
        opened = os.fstat(self.file_descriptor)
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise RuntimeError("Adaptive V4 GPU lock path disappeared while held.") from error
        expected_identity = (self.device, self.inode)
        _require(
            (opened.st_dev, opened.st_ino) == expected_identity
            and (current.st_dev, current.st_ino) == expected_identity,
            "Adaptive V4 GPU lock path was replaced while held.",
        )
        _require(
            stat.S_ISREG(opened.st_mode)
            and stat.S_ISREG(current.st_mode)
            and opened.st_uid == current.st_uid == os.getuid()
            and opened.st_nlink == current.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(current.st_mode) == SAFE_LOCK_MODE,
            "Adaptive V4 GPU lock metadata became unsafe while held.",
        )
        _require(
            opened.st_ctime_ns == current.st_ctime_ns == self.ctime_ns
            and opened.st_mtime_ns == current.st_mtime_ns == self.mtime_ns,
            "Adaptive V4 GPU lock metadata changed while held.",
        )
        observed = os.pread(self.file_descriptor, len(self.holder) + 1, 0)
        _require(observed == self.holder, "Adaptive V4 GPU lock owner record changed while held.")

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            self.assert_held()
        except BaseException as error:  # release the kernel lease even on fail-closed evidence
            failure = error
        try:
            fcntl.flock(self.file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.file_descriptor)
            self._closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> GPULockLease:
        self.assert_held()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def _canonical_lock_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    absolute.parent.mkdir(parents=True, exist_ok=True)
    _require(not absolute.is_symlink(), "Adaptive V4 GPU lock may not be a symbolic link.")
    try:
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as error:
        raise RuntimeError("Adaptive V4 GPU lock parent is inaccessible.") from error
    _require(
        resolved_parent == absolute.parent,
        "Adaptive V4 GPU lock parent must use its exact non-symlink path.",
    )
    return absolute


def _write_all(file_descriptor: int, payload: bytes) -> None:
    os.ftruncate(file_descriptor, 0)
    offset = 0
    while offset < len(payload):
        written = os.pwrite(file_descriptor, payload[offset:], offset)
        _require(written > 0, "Adaptive V4 GPU lock metadata write stalled.")
        offset += written
    os.fsync(file_descriptor)


def acquire_gpu_lock(label: str, *, path: Path = DEFAULT_LOCK_PATH) -> GPULockLease:
    """Acquire a non-blocking, persistent-inode lease for paper-grade GPU work."""

    _require(
        isinstance(label, str)
        and bool(label.strip())
        and label == label.strip()
        and "\n" not in label
        and "\r" not in label
        and len(label) <= 256,
        "Adaptive V4 GPU lock label is invalid.",
    )
    canonical = _canonical_lock_path(path)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Adaptive V4 GPU lock requires O_NOFOLLOW support.")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
    try:
        file_descriptor = os.open(canonical, flags, SAFE_LOCK_MODE)
    except OSError as error:
        raise RuntimeError("Adaptive V4 GPU lock could not be opened safely.") from error
    try:
        opened = os.fstat(file_descriptor)
        current = os.stat(canonical, follow_symlinks=False)
        _require(
            stat.S_ISREG(opened.st_mode)
            and stat.S_ISREG(current.st_mode)
            and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
            and opened.st_uid == current.st_uid == os.getuid()
            and opened.st_nlink == current.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(current.st_mode) == SAFE_LOCK_MODE,
            "Adaptive V4 GPU lock ownership, links, or mode are unsafe.",
        )
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            prior_holder = (
                os.pread(file_descriptor, 4096, 0).decode("utf-8", errors="replace").strip()
            )
            raise RuntimeError(
                f"Adaptive V4 GPU is already locked by {prior_holder or 'unknown holder'}."
            ) from error
        current = os.stat(canonical, follow_symlinks=False)
        opened = os.fstat(file_descriptor)
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Adaptive V4 GPU lock path changed during acquisition.",
        )
        active_holder = f"pid={os.getpid()} label={label}\n".encode()
        _write_all(file_descriptor, active_holder)
        opened = os.fstat(file_descriptor)
        lease = GPULockLease(
            path=canonical,
            file_descriptor=file_descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
            ctime_ns=opened.st_ctime_ns,
            mtime_ns=opened.st_mtime_ns,
            holder=active_holder,
        )
        lease.assert_held()
        return lease
    except BaseException:
        os.close(file_descriptor)
        raise


def acquire_device_guard(label: str, routing_identity: Mapping[str, Any]) -> GPULockLease:
    """Acquire the canonical second lease that cannot be bypassed with another user path."""

    path = canonical_device_guard_path(routing_identity)
    return acquire_gpu_lock(f"{label}:physical-device", path=path)
