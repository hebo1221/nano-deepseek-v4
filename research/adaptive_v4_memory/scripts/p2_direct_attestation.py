from __future__ import annotations

import builtins
import fcntl
import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

SCHEME = "hmac-sha256-v1"
KEY_FD_ENV = "ADAPTIVE_V4_ATTESTATION_KEY_FD"
KEY_PATH_ENV = "ADAPTIVE_V4_ATTESTATION_KEY_PATH"
MINIMUM_KEY_BYTES = 32
MAXIMUM_KEY_BYTES = 4096
THREAT_MODEL = (
    "authenticates artifacts produced by the trusted canonical runner and detects forged, "
    "concurrently substituted, or binding-mismatched stale files; compromise of the user-owned "
    "key, ptrace or memory inspection by the same user, a hostile process already holding the "
    "key, and whole-set rollback of an otherwise valid identically bound experiment are out of "
    "scope"
)
KEY_STORAGE_RULE = (
    "raw high-entropy key outside repository and artifact roots, owned by the invoking user, "
    "regular non-symlink file, mode 0600; key bytes never enter commands, artifacts, or logs"
)
KEY_TRANSPORT = "runner-to-child inherited sealed memfd"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def checksum(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def checksum_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checksum_fd(file_descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(file_descriptor, 1024 * 1024, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def read_all_fd(file_descriptor: int, *, maximum_bytes: int | None = None) -> bytes:
    opened = os.fstat(file_descriptor)
    _require(stat.S_ISREG(opened.st_mode), "Attested input must be a regular file.")
    if maximum_bytes is not None:
        _require(opened.st_size <= maximum_bytes, "Attested input exceeds its size limit.")
    chunks: list[bytes] = []
    offset = 0
    while offset < opened.st_size:
        chunk = os.pread(file_descriptor, min(1024 * 1024, opened.st_size - offset), offset)
        _require(bool(chunk), "Attested input was truncated while being read.")
        chunks.append(chunk)
        offset += len(chunk)
    after = os.fstat(file_descriptor)
    _require(
        (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "Attested input changed while being read.",
    )
    return b"".join(chunks)


@dataclass(frozen=True)
class OpenedRegularFile:
    path: Path
    file_descriptor: int
    sha256: str
    bytes: int
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    def close(self) -> None:
        os.close(self.file_descriptor)

    def read_bytes(self, *, maximum_bytes: int | None = None) -> builtins.bytes:
        return read_all_fd(self.file_descriptor, maximum_bytes=maximum_bytes)

    def duplicate_binary_handle(self) -> Any:
        duplicated = os.dup(self.file_descriptor)
        os.lseek(duplicated, 0, os.SEEK_SET)
        return os.fdopen(duplicated, "rb")

    def assert_unchanged(self) -> None:
        opened = os.fstat(self.file_descriptor)
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise ValueError("Attested path disappeared during validation.") from error
        expected = (
            self.device,
            self.inode,
            self.bytes,
            self.mode,
            self.mtime_ns,
            self.ctime_ns,
        )
        observed = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mode,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        _require(observed == expected, "Open attested file changed during validation.")
        _require(
            (current.st_dev, current.st_ino) == (self.device, self.inode),
            "Attested path was replaced during validation.",
        )
        _require(checksum_fd(self.file_descriptor) == self.sha256, "Attested bytes changed.")


def open_regular_nofollow(path: Path) -> OpenedRegularFile:
    absolute = Path(os.path.abspath(path))
    _require(not absolute.is_symlink(), "Attested path may not be a symbolic link.")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Attested file access requires O_NOFOLLOW support.")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
    try:
        file_descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValueError("Attested regular file could not be opened safely.") from error
    try:
        opened = os.fstat(file_descriptor)
        current = os.stat(absolute, follow_symlinks=False)
        _require(stat.S_ISREG(opened.st_mode), "Attested path is not a regular file.")
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Attested path changed while it was opened.",
        )
        return OpenedRegularFile(
            path=absolute,
            file_descriptor=file_descriptor,
            sha256=checksum_fd(file_descriptor),
            bytes=opened.st_size,
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=opened.st_mode,
            mtime_ns=opened.st_mtime_ns,
            ctime_ns=opened.st_ctime_ns,
        )
    except BaseException:
        os.close(file_descriptor)
        raise


@dataclass(frozen=True, repr=False)
class TrustRoot:
    key: bytes
    key_id: str

    def __post_init__(self) -> None:
        _require(type(self.key) is bytes, "Attestation key must be bytes.")
        _require(
            MINIMUM_KEY_BYTES <= len(self.key) <= MAXIMUM_KEY_BYTES,
            "Attestation key must contain at least 256 bits and stay within its size limit.",
        )
        _require(
            len(set(self.key)) >= 16,
            "Attestation key is structurally low-entropy; generate fresh random key bytes.",
        )
        _require(
            isinstance(self.key_id, str)
            and len(self.key_id) == 64
            and all(character in "0123456789abcdef" for character in self.key_id),
            "Attestation key ID is invalid.",
        )
        _require(
            hmac.compare_digest(derive_key_id(self.key), self.key_id),
            "Attestation key ID does not match its key material.",
        )

    def __repr__(self) -> str:
        return f"TrustRoot(key=<redacted>, key_id={self.key_id!r})"


def derive_key_id(key: bytes) -> str:
    return hashlib.sha256(b"adaptive-v4-attestation-key-id-v1\0" + key).hexdigest()


def _validated_key(key: bytes, *, expected_key_id: str | None = None) -> TrustRoot:
    _require(
        MINIMUM_KEY_BYTES <= len(key) <= MAXIMUM_KEY_BYTES,
        "Attestation key must contain at least 256 bits and stay within its size limit.",
    )
    _require(
        len(set(key)) >= 16,
        "Attestation key is structurally low-entropy; generate fresh random key bytes.",
    )
    key_id = derive_key_id(key)
    if expected_key_id is not None:
        _require(
            hmac.compare_digest(key_id, expected_key_id),
            "Attestation key does not match the frozen manifest key ID.",
        )
    return TrustRoot(key=key, key_id=key_id)


def load_trust_root(
    path: Path,
    *,
    repository_root: Path,
    artifact_roots: tuple[Path, ...] = (),
    expected_key_id: str | None = None,
) -> TrustRoot:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = absolute.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("Attestation trust root is missing or inaccessible.") from error
    forbidden = (repository, *(root.resolve(strict=False) for root in artifact_roots))
    _require(
        all(not resolved.is_relative_to(root) for root in forbidden),
        "Attestation key must live outside repository and artifact roots.",
    )
    opened = open_regular_nofollow(absolute)
    try:
        try:
            opened_resolved = Path(f"/proc/self/fd/{opened.file_descriptor}").resolve(strict=True)
        except OSError as error:
            raise ValueError("Attestation key open-file identity cannot be resolved.") from error
        _require(
            opened_resolved == resolved,
            "Attestation key path changed while the trust root was opened.",
        )
        _require(
            all(not opened_resolved.is_relative_to(root) for root in forbidden),
            "Opened attestation key must live outside repository and artifact roots.",
        )
        metadata = os.fstat(opened.file_descriptor)
        _require(metadata.st_uid == os.getuid(), "Attestation key must be owned by this user.")
        _require(stat.S_IMODE(metadata.st_mode) == 0o600, "Attestation key mode must be 0600.")
        _require(metadata.st_nlink == 1, "Attestation key must not have multiple hard links.")
        key = opened.read_bytes(maximum_bytes=MAXIMUM_KEY_BYTES)
        opened.assert_unchanged()
        return _validated_key(key, expected_key_id=expected_key_id)
    finally:
        opened.close()


def trust_root_from_environment(
    *,
    repository_root: Path,
    artifact_roots: tuple[Path, ...] = (),
    expected_key_id: str | None = None,
) -> TrustRoot:
    raw = os.environ.get(KEY_PATH_ENV)
    _require(bool(raw), f"{KEY_PATH_ENV} must identify the external attestation trust root.")
    return load_trust_root(
        Path(cast(str, raw)),
        repository_root=repository_root,
        artifact_roots=artifact_roots,
        expected_key_id=expected_key_id,
    )


def create_sealed_key_fd(trust_root: TrustRoot) -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(
        create is not None and allow_sealing is not None,
        "Attestation transport requires sealed memfd support.",
    )
    create_memfd = cast(Any, create)
    descriptor = create_memfd(
        "adaptive-v4-attestation-key",
        cast(int, getattr(os, "MFD_CLOEXEC", 0)) | cast(int, allow_sealing),
    )
    try:
        written = 0
        while written < len(trust_root.key):
            written += os.write(descriptor, trust_root.key[written:])
        required_seals = (
            fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, required_seals)
        _require(
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & required_seals == required_seals,
            "Attestation key memfd was not sealed.",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def trust_root_from_sealed_fd(
    file_descriptor: int,
    *,
    expected_key_id: str,
    close: bool = True,
) -> TrustRoot:
    try:
        seals = fcntl.fcntl(file_descriptor, fcntl.F_GET_SEALS)
        required = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        _require(seals & required == required, "Inherited attestation key FD is not sealed.")
        key = read_all_fd(file_descriptor, maximum_bytes=MAXIMUM_KEY_BYTES)
        return _validated_key(key, expected_key_id=expected_key_id)
    finally:
        if close:
            os.close(file_descriptor)


def trust_root_from_inherited_environment(*, expected_key_id: str) -> TrustRoot:
    raw = os.environ.pop(KEY_FD_ENV, None)
    _require(raw is not None and raw.isdigit(), "Inherited attestation key FD is missing.")
    return trust_root_from_sealed_fd(int(cast(str, raw)), expected_key_id=expected_key_id)


def attest_payload(
    payload: Mapping[str, Any],
    *,
    trust_root: TrustRoot,
    purpose: str,
) -> dict[str, str]:
    _require(bool(purpose), "Attestation purpose must be non-empty.")
    payload_sha256 = checksum(payload)
    authenticated = {
        "scheme": SCHEME,
        "key_id": trust_root.key_id,
        "purpose": purpose,
        "payload_sha256": payload_sha256,
    }
    mac = hmac.new(trust_root.key, canonical_json(authenticated), hashlib.sha256).hexdigest()
    return {**authenticated, "mac": mac}


def verify_attestation(
    payload: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    trust_root: TrustRoot,
    purpose: str,
) -> None:
    _require(
        set(envelope) == {"scheme", "key_id", "purpose", "payload_sha256", "mac"},
        "Attestation envelope schema drifted.",
    )
    _require(envelope.get("scheme") == SCHEME, "Attestation scheme drifted.")
    _require(envelope.get("key_id") == trust_root.key_id, "Attestation key ID drifted.")
    _require(envelope.get("purpose") == purpose, "Attestation purpose drifted.")
    payload_sha256 = checksum(payload)
    _require(
        envelope.get("payload_sha256") == payload_sha256,
        "Attestation payload checksum drifted.",
    )
    authenticated = {
        "scheme": SCHEME,
        "key_id": trust_root.key_id,
        "purpose": purpose,
        "payload_sha256": payload_sha256,
    }
    expected = hmac.new(trust_root.key, canonical_json(authenticated), hashlib.sha256).hexdigest()
    observed = envelope.get("mac")
    _require(
        isinstance(observed, str) and hmac.compare_digest(observed, expected),
        "Attestation MAC verification failed.",
    )


def public_manifest_contract(key_id: str) -> dict[str, Any]:
    _require(
        len(key_id) == 64 and all(character in "0123456789abcdef" for character in key_id),
        "Attestation key ID is invalid.",
    )
    return {
        "scheme": SCHEME,
        "key_id": key_id,
        "key_storage_rule": KEY_STORAGE_RULE,
        "key_transport": KEY_TRANSPORT,
        "threat_model": THREAT_MODEL,
        "key_compromise_in_scope": False,
    }
