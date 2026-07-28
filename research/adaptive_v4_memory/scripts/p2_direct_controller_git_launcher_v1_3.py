from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, cast

SCHEMA_VERSION = 1
LAUNCHER_ID = "p2-direct-controller-git-object-launcher-v1-3-5"
PINNED_GIT_EXECUTABLE = Path("/usr/bin/git")
MANIFEST_RELATIVE_PATH = (
    "research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-exact-fill-v1-3-5.json"
)
LAUNCHER_RELATIVE_PATH = (
    "research/adaptive_v4_memory/scripts/p2_direct_controller_git_launcher_v1_3.py"
)
RUNNER_RELATIVE_PATH = "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix_v1_3.py"
AUDIT_RELATIVE_PATH = (
    "research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity_v1_3.py"
)
SUMMARY_RELATIVE_PATH = "research/adaptive_v4_memory/scripts/summarize_p2_direct_controller_v1_3.py"
ENTRYPOINT_RELATIVE_PATHS = {
    "matrix": RUNNER_RELATIVE_PATH,
    "audit": AUDIT_RELATIVE_PATH,
    "summary": SUMMARY_RELATIVE_PATH,
}
PYTHON_RELATIVE_PATH = ".venv/bin/python"
EXPECTED_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-v1.3.5"
EXPECTED_MANIFEST_STATUS = (
    "frozen_v1_3_5_three_worker_parallel_final_after_live_claim_preflight_failure"
)
RUNNER_FD_ENV = "ADAPTIVE_V4_DIRECT_EXACT_FILL_V1_3_5_GIT_RUNNER_FD"
SOURCE_BUNDLE_FD_ENV = "ADAPTIVE_V4_DIRECT_EXACT_FILL_V1_3_5_GIT_SOURCE_BUNDLE_FD"
LAUNCH_ROUTING_FD_ENV = "ADAPTIVE_V4_DIRECT_EXACT_FILL_V1_3_5_GIT_LAUNCH_ROUTING_FD"
SEALED_LAUNCH_SENTINEL_NAME = "_ADAPTIVE_V4_GIT_OBJECT_LAUNCH_SENTINEL_V1_3_5"
PYTHON_RUNTIME_BINDING_NAME = "_ADAPTIVE_V4_GIT_OBJECT_PYTHON_RUNTIME_V1_3_5"
SOURCE_PROVENANCE_BINDING_NAME = "_ADAPTIVE_V4_GIT_OBJECT_SOURCE_PROVENANCE_V1_3_5"
LAUNCH_ROUTING_BINDING_NAME = "_ADAPTIVE_V4_GIT_OBJECT_LAUNCH_ROUTING_V1_3_5"
SEALED_LAUNCH_SENTINEL = {
    "schema_version": 1,
    "launcher": LAUNCHER_ID,
    "sealed_runner": True,
    "sealed_inventory": True,
}
MAXIMUM_SOURCE_BUNDLE_BYTES = 128 << 20
MAXIMUM_LAUNCH_ROUTING_BYTES = 16 << 10
MAXIMUM_PYTHON_EXECUTABLE_BYTES = 128 << 20
MAXIMUM_IMPLEMENTATION_FILES = 4_096
MAXIMUM_RAW_ANCESTRY_COMMITS = 100_000
_FULL_SEALS = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_object_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _json_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_relative_path(value: object, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is invalid.")
    path = PurePosixPath(cast(str, value))
    _require(
        not path.is_absolute()
        and ".." not in path.parts
        and "." not in path.parts
        and str(path) == value
        and "\\" not in value
        and "\x00" not in value,
        f"{label} is not a canonical repository-relative path.",
    )
    return cast(str, value)


def _entrypoint_relative_path(selector: object) -> str:
    _require(
        isinstance(selector, str) and selector in ENTRYPOINT_RELATIVE_PATHS,
        "Canonical Git-object entrypoint selector is not allowlisted.",
    )
    return ENTRYPOINT_RELATIVE_PATHS[cast(str, selector)]


def _exact_repository_root(path: Path) -> Path:
    _require(path.is_absolute(), "Canonical repository root must be absolute.")
    absolute = Path(os.path.abspath(path))
    resolved = absolute.resolve(strict=True)
    _require(
        absolute == resolved and resolved.is_dir() and not resolved.is_symlink(),
        "Canonical repository root is not exact.",
    )
    return resolved


def validate_pinned_git_executable(path: Path = PINNED_GIT_EXECUTABLE) -> Path:
    _require(path.is_absolute(), "Pinned Git executable path must be absolute.")
    absolute = Path(os.path.abspath(path))
    resolved = absolute.resolve(strict=True)
    metadata = os.stat(resolved, follow_symlinks=False)
    _require(
        absolute == resolved
        and stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == 0
        and metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH) == 0
        and os.access(resolved, os.X_OK),
        "Pinned Git executable metadata is unsafe.",
    )
    return resolved


def _python_environment() -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": "/nonexistent",
    }


def _git_environment() -> dict[str, str]:
    return {
        **_python_environment(),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _python_runtime_binding(repository_root: Path) -> dict[str, Any]:
    """Bind the active root-owned interpreter and one inert venv import directory."""

    root = _exact_repository_root(repository_root)
    _require(
        sys.implementation.name == "cpython" and sys.version_info >= (3, 10),
        "Canonical launcher requires CPython 3.10 or newer.",
    )
    venv_executable = Path(os.path.abspath(root / PYTHON_RELATIVE_PATH))
    executable_target = venv_executable.resolve(strict=True)
    active_target = Path("/proc/self/exe").resolve(strict=True)
    _require(
        executable_target == active_target,
        "Repository venv Python does not resolve to the active interpreter.",
    )
    metadata = os.stat(active_target, follow_symlinks=False)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == 0
        and metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH) == 0
        and 0 < metadata.st_size <= MAXIMUM_PYTHON_EXECUTABLE_BYTES
        and os.access(active_target, os.X_OK),
        "Active Python interpreter metadata is unsafe.",
    )
    digest = hashlib.sha256()
    observed_bytes = 0
    with Path("/proc/self/exe").open("rb") as stream:
        while chunk := stream.read(1 << 20):
            observed_bytes += len(chunk)
            _require(
                observed_bytes <= MAXIMUM_PYTHON_EXECUTABLE_BYTES,
                "Active Python interpreter exceeds its size limit.",
            )
            digest.update(chunk)
    _require(
        observed_bytes == metadata.st_size,
        "Active Python interpreter bytes changed while binding the runtime.",
    )
    version_directory = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(
        os.path.abspath(root / ".venv" / "lib" / version_directory / "site-packages")
    )
    resolved_site_packages = site_packages.resolve(strict=True)
    _require(
        site_packages == resolved_site_packages
        and resolved_site_packages.is_dir()
        and not resolved_site_packages.is_symlink(),
        "Repository venv site-packages directory is not exact.",
    )
    base_prefix = Path(sys.base_prefix).resolve(strict=True)
    return {
        "schema_version": 1,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
            sys.version_info.releaselevel,
            sys.version_info.serial,
        ],
        "executable": str(active_target),
        "venv_executable": str(venv_executable),
        "resolved_executable": str(active_target),
        "executable_sha256": digest.hexdigest(),
        "executable_metadata": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "bytes": metadata.st_size,
        },
        "base_prefix": str(base_prefix),
        "site_packages": str(site_packages),
    }


def _git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    executable = validate_pinned_git_executable()
    result = subprocess.run(
        [
            str(executable),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.commitGraph=false",
            "-C",
            str(repository_root),
            *arguments,
        ],
        check=False,
        capture_output=True,
        env=_git_environment(),
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Pinned Git object query failed: {detail}")
    return result


def pin_starting_head(repository_root: Path, expected_head_oid: str) -> str:
    root = _exact_repository_root(repository_root)
    _require(_is_object_id(expected_head_oid), "Starting HEAD object ID is invalid.")
    observed = _git(root, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout.decode("ascii").strip()
    _require(
        observed == expected_head_oid,
        "Starting HEAD changed before the canonical launcher pinned it.",
    )
    _raw_commit_parents(root, expected_head_oid)
    return expected_head_oid


def _raw_git_object(repository_root: Path, object_id: str, object_type: str) -> bytes:
    _require(_is_object_id(object_id), f"Git {object_type} object ID is invalid.")
    _require(object_type in {"blob", "commit"}, "Unsupported raw Git object type.")
    data = _git(repository_root, ["cat-file", object_type, object_id]).stdout
    algorithm = "sha1" if len(object_id) == 40 else "sha256"
    source = object_type.encode("ascii") + b" " + str(len(data)).encode("ascii") + b"\0" + data
    _require(
        hashlib.new(algorithm, source).hexdigest() == object_id,
        f"Pinned Git {object_type} bytes do not reproduce their object ID.",
    )
    return data


def _raw_commit_parents(repository_root: Path, commit_oid: str) -> tuple[str, ...]:
    data = _raw_git_object(repository_root, commit_oid, "commit")
    headers, separator, _message = data.partition(b"\n\n")
    _require(separator == b"\n\n", "Pinned Git commit object lacks a header boundary.")
    tree_count = 0
    parents: list[str] = []
    for line in headers.splitlines():
        if line.startswith(b"tree "):
            tree_count += 1
            tree_oid = line.removeprefix(b"tree ").decode("ascii")
            _require(_is_object_id(tree_oid), "Pinned Git commit tree ID is invalid.")
        elif line.startswith(b"parent "):
            parent_oid = line.removeprefix(b"parent ").decode("ascii")
            _require(
                _is_object_id(parent_oid) and len(parent_oid) == len(commit_oid),
                "Pinned Git commit parent ID is invalid.",
            )
            parents.append(parent_oid)
    _require(tree_count == 1, "Pinned Git commit tree header is invalid.")
    return tuple(parents)


def _raw_commit_is_ancestor(repository_root: Path, ancestor_oid: str, descendant_oid: str) -> bool:
    _require(
        _is_object_id(ancestor_oid)
        and _is_object_id(descendant_oid)
        and len(ancestor_oid) == len(descendant_oid),
        "Raw Git ancestry object IDs are invalid.",
    )
    pending = [descendant_oid]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        _require(
            len(visited) <= MAXIMUM_RAW_ANCESTRY_COMMITS,
            "Raw Git ancestry exceeds its commit bound.",
        )
        if current == ancestor_oid:
            return True
        pending.extend(
            parent
            for parent in _raw_commit_parents(repository_root, current)
            if parent not in visited
        )
    return False


def _tree_rows(
    repository_root: Path,
    revision: str,
    pathspecs: Sequence[str],
) -> tuple[dict[str, str], ...]:
    _require(_is_object_id(revision), "Git tree revision is invalid.")
    canonical_specs = tuple(
        _canonical_relative_path(path, label="Implementation pathspec") for path in pathspecs
    )
    _require(
        bool(canonical_specs) and len(canonical_specs) == len(set(canonical_specs)),
        "Implementation pathspec inventory is empty or duplicated.",
    )
    raw = _git(
        repository_root,
        ["ls-tree", "-r", "-z", "--full-tree", revision, "--", *canonical_specs],
    ).stdout
    rows: list[dict[str, str]] = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, separator, encoded_path = entry.partition(b"\t")
        fields = metadata.split()
        _require(
            separator == b"\t" and len(fields) == 3,
            "Git tree inventory entry is malformed.",
        )
        mode, object_type, object_id = (field.decode("ascii") for field in fields)
        path = _canonical_relative_path(
            encoded_path.decode("utf-8"), label="Git tree inventory path"
        )
        _require(
            mode in {"100644", "100755"} and object_type == "blob" and _is_object_id(object_id),
            "Git implementation inventory contains a non-regular blob.",
        )
        rows.append(
            {
                "path": path,
                "git_mode": mode,
                "git_blob_oid": object_id,
            }
        )
    _require(
        1 <= len(rows) <= MAXIMUM_IMPLEMENTATION_FILES
        and [row["path"] for row in rows] == sorted({row["path"] for row in rows}),
        "Git implementation inventory is empty, duplicated, or non-canonical.",
    )
    observed_paths = tuple(row["path"] for row in rows)
    for pathspec in canonical_specs:
        prefix = pathspec.rstrip("/") + "/"
        _require(
            pathspec in observed_paths or any(path.startswith(prefix) for path in observed_paths),
            f"Frozen implementation pathspec did not resolve to a blob: {pathspec}",
        )
    return tuple(rows)


def _single_tree_row(repository_root: Path, revision: str, relative_path: str) -> dict[str, str]:
    rows = _tree_rows(repository_root, revision, (relative_path,))
    _require(
        len(rows) == 1 and rows[0]["path"] == relative_path,
        f"Expected one exact Git blob at {relative_path}.",
    )
    return dict(rows[0])


def _git_blob(repository_root: Path, object_id: str) -> bytes:
    return _raw_git_object(repository_root, object_id, "blob")


def _implementation_index_digest(
    implementation_paths: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> str:
    entries = [f"{row['git_mode']} {row['git_blob_oid']} 0\t{row['path']}" for row in rows]
    return _json_digest(
        {
            "schema_version": 1,
            "implementation_paths": list(implementation_paths),
            "git_index_entries": entries,
        }
    )


def _load_head_manifest(
    repository_root: Path,
    head_oid: str,
    *,
    manifest_relative_path: str,
    expected_experiment_id: str,
    expected_manifest_status: str,
) -> tuple[dict[str, Any], dict[str, str], bytes]:
    path = _canonical_relative_path(manifest_relative_path, label="Manifest path")
    row = _single_tree_row(repository_root, head_oid, path)
    data = _git_blob(repository_root, row["git_blob_oid"])
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("HEAD manifest blob is not valid JSON.") from error
    _require(isinstance(payload, dict), "HEAD manifest blob is not an object.")
    canonical_pretty = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _require(data == canonical_pretty, "HEAD manifest blob is not canonical pretty JSON.")
    _require(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("experiment_id") == expected_experiment_id
        and payload.get("status") == expected_manifest_status,
        "HEAD manifest identity or freeze status drifted.",
    )
    return cast(dict[str, Any], payload), row, data


def build_frozen_source_bundle(
    repository_root: Path,
    pinned_head_oid: str,
    *,
    manifest_relative_path: str = MANIFEST_RELATIVE_PATH,
    launcher_relative_path: str = LAUNCHER_RELATIVE_PATH,
    runner_relative_path: str = RUNNER_RELATIVE_PATH,
    expected_experiment_id: str = EXPECTED_EXPERIMENT_ID,
    expected_manifest_status: str = EXPECTED_MANIFEST_STATUS,
) -> tuple[dict[str, Any], bytes]:
    """Build only from pinned Git objects; live worktree source bytes are never read."""

    root = _exact_repository_root(repository_root)
    _require(_is_object_id(pinned_head_oid), "Pinned HEAD object ID is invalid.")
    launcher_path = _canonical_relative_path(launcher_relative_path, label="Launcher path")
    runner_path = _canonical_relative_path(runner_relative_path, label="Runner path")
    entrypoint_paths = {
        selector: _canonical_relative_path(path, label=f"{selector} entrypoint path")
        for selector, path in ENTRYPOINT_RELATIVE_PATHS.items()
    }
    _require(
        runner_path in entrypoint_paths.values(),
        "Frozen Git-object entrypoint path is not allowlisted.",
    )
    manifest, manifest_row, manifest_bytes = _load_head_manifest(
        root,
        pinned_head_oid,
        manifest_relative_path=manifest_relative_path,
        expected_experiment_id=expected_experiment_id,
        expected_manifest_status=expected_manifest_status,
    )
    implementation = manifest.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and set(implementation) == {"paths", "tree_digest", "source_commit"},
        "HEAD manifest implementation binding is invalid.",
    )
    raw_paths = cast(Mapping[str, Any], implementation).get("paths")
    _require(isinstance(raw_paths, list), "Manifest implementation paths are invalid.")
    implementation_paths = tuple(
        _canonical_relative_path(path, label="Manifest implementation path")
        for path in cast(list[Any], raw_paths)
    )
    _require(
        bool(implementation_paths) and len(implementation_paths) == len(set(implementation_paths)),
        "Manifest implementation path inventory is empty or duplicated.",
    )
    source_commit = cast(Mapping[str, Any], implementation).get("source_commit")
    tree_digest = cast(Mapping[str, Any], implementation).get("tree_digest")
    _require(
        _is_object_id(source_commit) and _is_sha256(tree_digest),
        "Manifest implementation commit or digest is invalid.",
    )
    source_commit = cast(str, source_commit)
    _require(
        _raw_commit_is_ancestor(root, source_commit, pinned_head_oid),
        "Frozen implementation source commit is not an ancestor of pinned HEAD.",
    )
    rows = _tree_rows(root, source_commit, implementation_paths)
    _require(
        _implementation_index_digest(implementation_paths, rows) == tree_digest,
        "Frozen implementation tree digest differs from the HEAD manifest.",
    )
    row_by_path = {row["path"]: row for row in rows}
    protected_paths = (launcher_path, *entrypoint_paths.values())
    _require(
        all(path in row_by_path for path in protected_paths),
        "Frozen implementation omits the canonical launcher or an allowlisted entrypoint.",
    )
    for protected_path in protected_paths:
        head_row = _single_tree_row(root, pinned_head_oid, protected_path)
        _require(
            head_row == row_by_path[protected_path],
            f"Pinned HEAD changed frozen executable bytes: {protected_path}",
        )

    files: list[dict[str, Any]] = []
    runner_bytes: bytes | None = None
    for row in rows:
        data = _git_blob(root, row["git_blob_oid"])
        record = {
            **row,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "source_base64": base64.b64encode(data).decode("ascii"),
        }
        files.append(record)
        if row["path"] == runner_path:
            runner_bytes = data
    _require(runner_bytes is not None, "Frozen runner bytes are unavailable.")
    source = {
        "schema_version": SCHEMA_VERSION,
        "launcher": LAUNCHER_ID,
        "repository_root": str(root),
        "python_runtime": _python_runtime_binding(root),
        "pinned_head_oid": pinned_head_oid,
        "head_manifest": {
            "path": manifest_relative_path,
            "git_mode": manifest_row["git_mode"],
            "git_blob_oid": manifest_row["git_blob_oid"],
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "bytes": len(manifest_bytes),
            "source_base64": base64.b64encode(manifest_bytes).decode("ascii"),
        },
        "frozen_source_commit": source_commit,
        "implementation_tree_digest": tree_digest,
        "implementation_paths": list(implementation_paths),
        "launcher_relative_path": launcher_path,
        "entrypoint_relative_paths": entrypoint_paths,
        "files": files,
    }
    bundle = {**source, "bundle_sha256": _json_digest(source)}
    _require(
        len(_canonical_json(bundle)) <= MAXIMUM_SOURCE_BUNDLE_BYTES,
        "Frozen implementation source bundle exceeds its size limit.",
    )
    return bundle, cast(bytes, runner_bytes)


def _launch_routing_binding(
    bundle: Mapping[str, Any], runner_bytes: bytes, entrypoint: str
) -> dict[str, Any]:
    """Bind one invocation without changing the selector-invariant source authority."""

    relative_path = _entrypoint_relative_path(entrypoint)
    _require(
        bundle.get("entrypoint_relative_paths") == ENTRYPOINT_RELATIVE_PATHS,
        "Frozen source bundle entrypoint inventory drifted before launch.",
    )
    source_bundle_sha256 = bundle.get("bundle_sha256")
    unsigned_bundle = dict(bundle)
    unsigned_bundle.pop("bundle_sha256", None)
    _require(
        _is_sha256(source_bundle_sha256) and _json_digest(unsigned_bundle) == source_bundle_sha256,
        "Frozen source bundle digest drifted before routing.",
    )
    files = bundle.get("files")
    _require(isinstance(files, list), "Frozen implementation file inventory is invalid.")
    matching = [
        row
        for row in cast(list[Any], files)
        if isinstance(row, Mapping) and row.get("path") == relative_path
    ]
    _require(
        len(matching) == 1,
        "Frozen invocation entrypoint is missing or duplicated in the source bundle.",
    )
    row = cast(Mapping[str, Any], matching[0])
    _require(
        set(row) == {"path", "git_mode", "git_blob_oid", "sha256", "bytes", "source_base64"},
        "Frozen invocation entrypoint binding schema drifted.",
    )
    try:
        bundled_source = base64.b64decode(row["source_base64"], validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Frozen invocation entrypoint source encoding drifted.") from error
    _require(
        row.get("git_mode") in {"100644", "100755"}
        and _is_object_id(row.get("git_blob_oid"))
        and _is_sha256(row.get("sha256"))
        and type(row.get("bytes")) is int
        and row.get("bytes") == len(runner_bytes)
        and row.get("sha256") == hashlib.sha256(runner_bytes).hexdigest()
        and bundled_source == runner_bytes,
        "Frozen invocation entrypoint bytes drifted before routing.",
    )
    routing = {
        "schema_version": SCHEMA_VERSION,
        "launcher": LAUNCHER_ID,
        "entrypoint_selector": entrypoint,
        "entrypoint_relative_path": relative_path,
        "source_bundle_sha256": source_bundle_sha256,
        "git_mode": row["git_mode"],
        "git_blob_oid": row["git_blob_oid"],
        "sha256": row["sha256"],
        "bytes": row["bytes"],
    }
    _require(
        len(_canonical_json(routing)) <= MAXIMUM_LAUNCH_ROUTING_BYTES,
        "Frozen invocation routing binding exceeds its size limit.",
    )
    return routing


def _create_sealed_memfd(name: str, data: bytes) -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(
        callable(create) and allow_sealing is not None,
        "Canonical Git-object launch requires sealed memfd support.",
    )
    descriptor = cast(
        int,
        cast(Any, create)(
            name,
            cast(int, getattr(os, "MFD_CLOEXEC", 0)) | cast(int, allow_sealing),
        ),
    )
    try:
        offset = 0
        while offset < len(data):
            count = os.write(descriptor, data[offset:])
            _require(count > 0, "Sealed Git-object transport write stalled.")
            offset += count
        _require(
            os.pread(descriptor, len(data) + 1, 0) == data,
            "Sealed Git-object transport bytes drifted.",
        )
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, _FULL_SEALS)
        _require(
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & _FULL_SEALS == _FULL_SEALS,
            "Git-object transport descriptor was not fully sealed.",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


RUNNER_BOOTSTRAP_SOURCE = r"""
import base64
import fcntl
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import stat
import sys

RUNNER_FD_ENV = "ADAPTIVE_V4_DIRECT_EXACT_FILL_V1_3_5_GIT_RUNNER_FD"
BUNDLE_FD_ENV = "ADAPTIVE_V4_DIRECT_EXACT_FILL_V1_3_5_GIT_SOURCE_BUNDLE_FD"
ROUTING_FD_ENV = "ADAPTIVE_V4_DIRECT_EXACT_FILL_V1_3_5_GIT_LAUNCH_ROUTING_FD"
FULL_SEALS = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
MAXIMUM_SOURCE_BUNDLE_BYTES = 128 << 20
MAXIMUM_LAUNCH_ROUTING_BYTES = 16 << 10
MAXIMUM_PYTHON_EXECUTABLE_BYTES = 128 << 20


def require(condition, message):
    if not condition:
        raise ImportError(message)


def read_sealed(fd, limit, label):
    metadata = os.fstat(fd)
    require(stat.S_ISREG(metadata.st_mode) and 0 <= metadata.st_size <= limit, f"{label} FD metadata drifted.")
    try:
        seals = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
    except OSError as error:
        raise ImportError(f"{label} FD is not sealable.") from error
    require(seals & FULL_SEALS == FULL_SEALS, f"{label} FD is not fully sealed.")
    data = os.pread(fd, metadata.st_size + 1, 0)
    require(len(data) == metadata.st_size, f"{label} FD bytes drifted.")
    return data


require(
    sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode,
    "Frozen runner bootstrap requires isolated, no-site, no-bytecode Python.",
)
runner_fd = int(os.environ[RUNNER_FD_ENV])
bundle_fd = int(os.environ[BUNDLE_FD_ENV])
routing_fd = int(os.environ[ROUTING_FD_ENV])
require(
    min(runner_fd, bundle_fd, routing_fd) >= 0
    and len({runner_fd, bundle_fd, routing_fd}) == 3,
    "Frozen Git-object transport descriptors are invalid or aliased.",
)
try:
    bundle_bytes = read_sealed(bundle_fd, MAXIMUM_SOURCE_BUNDLE_BYTES, "Frozen source bundle")
    routing_bytes = read_sealed(routing_fd, MAXIMUM_LAUNCH_ROUTING_BYTES, "Frozen launch routing")
    runner_bytes = read_sealed(runner_fd, MAXIMUM_SOURCE_BUNDLE_BYTES, "Frozen runner")
finally:
    os.close(routing_fd)
    os.close(bundle_fd)
    os.close(runner_fd)
try:
    bundle = json.loads(bundle_bytes)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ImportError("Frozen source bundle is invalid JSON.") from error
require(isinstance(bundle, dict), "Frozen source bundle is not an object.")
require(
    json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") == bundle_bytes,
    "Frozen source bundle is not canonical JSON.",
)
try:
    routing = json.loads(routing_bytes)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    raise ImportError("Frozen launch routing is invalid JSON.") from error
require(isinstance(routing, dict), "Frozen launch routing is not an object.")
require(
    json.dumps(routing, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii") == routing_bytes,
    "Frozen launch routing is not canonical JSON.",
)
require(
    set(bundle) == {
        "schema_version", "launcher", "repository_root", "python_runtime", "pinned_head_oid", "head_manifest",
        "frozen_source_commit", "implementation_tree_digest", "implementation_paths",
        "launcher_relative_path", "entrypoint_relative_paths", "files", "bundle_sha256",
    },
    "Frozen source bundle schema drifted.",
)
require(
    set(routing) == {
        "schema_version", "launcher", "entrypoint_selector", "entrypoint_relative_path",
        "source_bundle_sha256", "git_mode", "git_blob_oid", "sha256", "bytes",
    },
    "Frozen launch routing schema drifted.",
)
unsigned_bundle = dict(bundle)
observed_digest = unsigned_bundle.pop("bundle_sha256")
require(
    isinstance(observed_digest, str)
    and hashlib.sha256(
        json.dumps(unsigned_bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest() == observed_digest,
    "Frozen source bundle digest drifted.",
)
require(
    bundle["schema_version"] == 1
    and bundle["launcher"] == "p2-direct-controller-git-object-launcher-v1-3-5",
    "Frozen source bundle identity drifted.",
)
entrypoint_paths = {
    "matrix": "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix_v1_3.py",
    "audit": "research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity_v1_3.py",
    "summary": "research/adaptive_v4_memory/scripts/summarize_p2_direct_controller_v1_3.py",
}
require(
    bundle["entrypoint_relative_paths"] == entrypoint_paths,
    "Frozen source bundle entrypoint inventory drifted.",
)
entrypoint_selector = routing["entrypoint_selector"]
runner_relative = routing["entrypoint_relative_path"]
require(
    routing["schema_version"] == 1
    and routing["launcher"] == bundle["launcher"]
    and routing["source_bundle_sha256"] == bundle["bundle_sha256"]
    and isinstance(entrypoint_selector, str)
    and entrypoint_selector in entrypoint_paths
    and runner_relative == entrypoint_paths[entrypoint_selector],
    "Frozen launch routing identity or source authority drifted.",
)
require(isinstance(bundle["repository_root"], str), "Frozen repository root is invalid.")
root = os.path.abspath(bundle["repository_root"])
require(root == os.path.realpath(root) and os.path.isdir(root), "Frozen repository root is not exact.")
require(
    os.path.abspath(os.getcwd()) == root and os.path.realpath(os.getcwd()) == root,
    "Frozen runner working directory differs from the exact repository root.",
)


def active_python_runtime_binding(repository_root):
    require(
        sys.implementation.name == "cpython" and sys.version_info >= (3, 10),
        "Frozen runner bootstrap requires CPython 3.10 or newer.",
    )
    venv_executable = os.path.abspath(
        os.path.join(repository_root, ".venv/bin/python")
    )
    active_target = os.path.realpath("/proc/self/exe")
    require(
        os.path.abspath(sys.executable) == active_target,
        "Frozen runner bootstrap used a different exact interpreter executable.",
    )
    require(
        os.path.realpath(venv_executable) == active_target
        and os.path.isfile(active_target),
        "Frozen runner bootstrap interpreter target drifted.",
    )
    metadata = os.stat(active_target, follow_symlinks=False)
    require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == 0
        and metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH) == 0
        and 0 < metadata.st_size <= MAXIMUM_PYTHON_EXECUTABLE_BYTES
        and os.access(active_target, os.X_OK),
        "Frozen runner bootstrap interpreter metadata is unsafe.",
    )
    digest = hashlib.sha256()
    observed_bytes = 0
    with open("/proc/self/exe", "rb") as stream:
        while True:
            chunk = stream.read(1 << 20)
            if not chunk:
                break
            observed_bytes += len(chunk)
            require(
                observed_bytes <= MAXIMUM_PYTHON_EXECUTABLE_BYTES,
                "Frozen runner bootstrap interpreter exceeds its size limit.",
            )
            digest.update(chunk)
    require(
        observed_bytes == metadata.st_size,
        "Frozen runner bootstrap interpreter bytes changed while validating it.",
    )
    version_directory = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = os.path.abspath(
        os.path.join(repository_root, ".venv", "lib", version_directory, "site-packages")
    )
    require(
        site_packages == os.path.realpath(site_packages) and os.path.isdir(site_packages),
        "Frozen runner bootstrap site-packages directory is not exact.",
    )
    base_prefix = os.path.realpath(sys.base_prefix)
    require(os.path.isdir(base_prefix), "Frozen runner bootstrap base prefix is invalid.")
    return {
        "schema_version": 1,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
            sys.version_info.releaselevel,
            sys.version_info.serial,
        ],
        "executable": active_target,
        "venv_executable": venv_executable,
        "resolved_executable": active_target,
        "executable_sha256": digest.hexdigest(),
        "executable_metadata": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "bytes": metadata.st_size,
        },
        "base_prefix": base_prefix,
        "site_packages": site_packages,
    }


python_runtime = bundle["python_runtime"]
require(
    isinstance(python_runtime, dict)
    and python_runtime == active_python_runtime_binding(root),
    "Frozen Python runtime binding drifted.",
)
site_packages = python_runtime["site_packages"]
head_manifest = bundle["head_manifest"]
require(
    isinstance(head_manifest, dict)
    and set(head_manifest)
    == {"path", "git_mode", "git_blob_oid", "sha256", "bytes", "source_base64"},
    "Frozen HEAD manifest binding schema drifted.",
)
manifest_path = head_manifest["path"]
manifest_oid = head_manifest["git_blob_oid"]
manifest_sha256 = head_manifest["sha256"]
manifest_size = head_manifest["bytes"]
require(
    isinstance(manifest_path, str)
    and manifest_path
    and not os.path.isabs(manifest_path)
    and os.path.normpath(manifest_path) == manifest_path
    and "\\" not in manifest_path
    and head_manifest["git_mode"] in {"100644", "100755"}
    and isinstance(manifest_oid, str)
    and len(manifest_oid) in {40, 64}
    and all(character in "0123456789abcdef" for character in manifest_oid)
    and isinstance(manifest_sha256, str)
    and len(manifest_sha256) == 64
    and all(character in "0123456789abcdef" for character in manifest_sha256)
    and type(manifest_size) is int
    and manifest_size >= 0
    and isinstance(head_manifest["source_base64"], str),
    "Frozen HEAD manifest binding is invalid.",
)
try:
    manifest_source = base64.b64decode(head_manifest["source_base64"], validate=True)
except (ValueError, TypeError) as error:
    raise ImportError("Frozen HEAD manifest source encoding drifted.") from error
manifest_blob = b"blob " + str(len(manifest_source)).encode("ascii") + b"\0" + manifest_source
require(
    len(manifest_source) == manifest_size
    and hashlib.sha256(manifest_source).hexdigest() == manifest_sha256
    and hashlib.new(
        "sha1" if len(manifest_oid) == 40 else "sha256", manifest_blob
    ).hexdigest()
    == manifest_oid,
    "Frozen HEAD manifest source bytes drifted.",
)
for oid_label in ("pinned_head_oid", "frozen_source_commit"):
    oid = bundle[oid_label]
    require(
        isinstance(oid, str)
        and len(oid) in {40, 64}
        and all(character in "0123456789abcdef" for character in oid),
        f"Frozen source provenance {oid_label} is invalid.",
    )
tree_digest = bundle["implementation_tree_digest"]
require(
    isinstance(tree_digest, str)
    and len(tree_digest) == 64
    and all(character in "0123456789abcdef" for character in tree_digest),
    "Frozen implementation tree digest is invalid.",
)
source_provenance = {
    "schema_version": 1,
    "launcher": bundle["launcher"],
    "repository_root": bundle["repository_root"],
    "bundle_sha256": bundle["bundle_sha256"],
    "pinned_head_oid": bundle["pinned_head_oid"],
    "frozen_source_commit": bundle["frozen_source_commit"],
    "implementation_tree_digest": bundle["implementation_tree_digest"],
    "head_manifest": dict(bundle["head_manifest"]),
}
files = bundle["files"]
require(isinstance(files, list) and files, "Frozen implementation file inventory is empty.")
paths = [row.get("path") if isinstance(row, dict) else None for row in files]
require(paths == sorted(set(paths)), "Frozen implementation file inventory is not canonical.")
frozen_sources = {}
frozen_bindings = {}
for row in files:
    require(
        set(row) == {"path", "git_mode", "git_blob_oid", "sha256", "bytes", "source_base64"},
        "Frozen implementation file record schema drifted.",
    )
    relative = row["path"]
    require(
        isinstance(relative, str)
        and relative
        and not os.path.isabs(relative)
        and os.path.normpath(relative) == relative
        and "\\" not in relative,
        "Frozen implementation path is unsafe.",
    )
    candidate = os.path.abspath(os.path.join(root, relative))
    require(os.path.commonpath((candidate, root)) == root, "Frozen implementation path escaped the repository.")
    git_mode = row["git_mode"]
    git_oid = row["git_blob_oid"]
    sha256 = row["sha256"]
    size = row["bytes"]
    require(
        git_mode in {"100644", "100755"}
        and isinstance(git_oid, str)
        and len(git_oid) in {40, 64}
        and all(character in "0123456789abcdef" for character in git_oid)
        and isinstance(sha256, str)
        and len(sha256) == 64
        and all(character in "0123456789abcdef" for character in sha256)
        and type(size) is int
        and size >= 0
        and isinstance(row["source_base64"], str),
        "Frozen implementation file binding is invalid.",
    )
    try:
        source = base64.b64decode(row["source_base64"], validate=True)
    except (ValueError, TypeError) as error:
        raise ImportError(f"Frozen implementation source encoding drifted: {relative}") from error
    blob_source = b"blob " + str(len(source)).encode("ascii") + b"\0" + source
    require(
        len(source) == size
        and hashlib.sha256(source).hexdigest() == sha256
        and hashlib.new("sha1" if len(git_oid) == 40 else "sha256", blob_source).hexdigest() == git_oid,
        f"Frozen implementation source bytes drifted: {relative}",
    )
    frozen_bindings[relative] = {
        "git_mode": git_mode,
        "git_blob_oid": git_oid,
        "sha256": sha256,
        "bytes": size,
    }
    if candidate.endswith(".py"):
        frozen_sources[candidate] = source

require(
    all(relative in frozen_bindings for relative in entrypoint_paths.values()),
    "Frozen source bundle omits an allowlisted entrypoint binding.",
)
runner_binding = frozen_bindings.get(runner_relative)
require(
    isinstance(runner_binding, dict)
    and routing
    == {
        "schema_version": 1,
        "launcher": bundle["launcher"],
        "entrypoint_selector": entrypoint_selector,
        "entrypoint_relative_path": runner_relative,
        "source_bundle_sha256": bundle["bundle_sha256"],
        **runner_binding,
    },
    "Frozen launch routing differs from its selected Git blob.",
)
runner_path = os.path.abspath(os.path.join(root, runner_relative))
require(
    runner_path in frozen_sources and frozen_sources[runner_path] == runner_bytes,
    "Sealed runner differs from its frozen Git blob.",
)
script_directory = os.path.dirname(runner_path)
environment_roots = tuple(
    sorted(
        {
            os.path.realpath(prefix)
            for prefix in (sys.prefix, sys.base_prefix, site_packages)
            if isinstance(prefix, str) and prefix
        }
    )
)


def is_within(candidate, directory):
    try:
        return os.path.commonpath((candidate, directory)) == directory
    except ValueError:
        return False


frozen_modules = {}


def register_frozen_import_root(import_root):
    package_initializers = {
        os.path.dirname(candidate)
        for candidate in frozen_sources
        if os.path.basename(candidate) == "__init__.py"
        and is_within(candidate, import_root)
    }
    for candidate, source in frozen_sources.items():
        if not is_within(candidate, import_root):
            continue
        relative = os.path.relpath(candidate, import_root)
        components = relative.split(os.sep)
        filename = components[-1]
        if not filename.endswith(".py"):
            continue
        package_components = components[:-1]
        package_chain = [
            os.path.join(import_root, *package_components[:index])
            for index in range(1, len(package_components) + 1)
        ]
        if any(directory not in package_initializers for directory in package_chain):
            continue
        if filename == "__init__.py":
            if not package_components:
                continue
            fullname = ".".join(package_components)
            is_package = True
        else:
            fullname = ".".join([*package_components, filename.removesuffix(".py")])
            is_package = False
        previous = frozen_modules.get(fullname)
        record = (candidate, source, is_package)
        require(
            previous is None or previous == record,
            f"Frozen Git module name is ambiguous: {fullname}",
        )
        frozen_modules[fullname] = record


register_frozen_import_root(root)
register_frozen_import_root(script_directory)
require(frozen_modules, "Frozen Git module inventory is empty.")
protected_top_levels = {fullname.partition(".")[0] for fullname in frozen_modules}
for fullname in frozen_modules:
    require(
        fullname not in sys.modules,
        f"Frozen Git module was imported before its sealed loader: {fullname}",
    )

for module in tuple(sys.modules.values()):
    origin = getattr(module, "__file__", None)
    if not isinstance(origin, str):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
    if not isinstance(origin, str) or origin in {"built-in", "frozen"}:
        continue
    candidate = os.path.abspath(origin)
    resolved = os.path.realpath(candidate)
    inside_environment = any(
        is_within(candidate, prefix) and is_within(resolved, prefix)
        for prefix in environment_roots
    )
    require(
        inside_environment or (not is_within(candidate, root) and not is_within(resolved, root)),
        f"Repository module was imported before the exclusive loader: {candidate}",
    )


class FrozenGitSourceLoader(importlib.abc.Loader):
    def __init__(self, fullname, origin, source):
        self.fullname = fullname
        self.origin = origin
        self.source = source

    def create_module(self, spec):
        del spec
        return None

    def exec_module(self, module):
        spec = getattr(module, "__spec__", None)
        require(
            getattr(module, "__file__", None) == self.origin
            and spec is not None
            and getattr(spec, "origin", None) == self.origin
            and getattr(spec, "loader", None) is self,
            f"Frozen Git module provenance drifted before execution: {self.fullname}",
        )
        code = compile(self.source, self.origin, "exec", dont_inherit=True)
        exec(code, module.__dict__, module.__dict__)


class ExclusiveRepositoryFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del target
        record = frozen_modules.get(fullname)
        if record is not None:
            candidate, source, is_package = record
            loader = FrozenGitSourceLoader(fullname, candidate, source)
            frozen = importlib.machinery.ModuleSpec(
                fullname, loader, origin=candidate, is_package=is_package
            )
            frozen.has_location = True
            if is_package:
                frozen.submodule_search_locations = [os.path.dirname(candidate)]
            return frozen
        if fullname.partition(".")[0] in protected_top_levels:
            raise ImportError(f"Blocked unresolved frozen Git import: {fullname}")
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            return spec
        if not isinstance(origin, str):
            locations = getattr(spec, "submodule_search_locations", None)
            if locations is not None:
                for location in locations:
                    candidate = os.path.abspath(location)
                    resolved = os.path.realpath(candidate)
                    inside_environment = any(
                        is_within(candidate, prefix) and is_within(resolved, prefix)
                        for prefix in environment_roots
                    )
                    if not inside_environment and (
                        is_within(candidate, root) or is_within(resolved, root)
                    ):
                        raise ImportError(
                            f"Blocked non-frozen repository namespace before execution: {fullname} -> {candidate}"
                        )
            return spec
        candidate = os.path.abspath(origin)
        resolved = os.path.realpath(candidate)
        inside_environment = any(
            is_within(candidate, prefix) and is_within(resolved, prefix)
            for prefix in environment_roots
        )
        if inside_environment:
            return spec
        if not is_within(candidate, root) and not is_within(resolved, root):
            return spec
        raise ImportError(
            f"Blocked non-frozen repository import before execution: {fullname} -> {candidate}"
        )


sys.meta_path.insert(0, ExclusiveRepositoryFinder())
sys.path.insert(0, root)
sys.path.insert(0, script_directory)
sys.path.append(site_packages)
sys.argv = [runner_path, *sys.argv[1:]]
globals_dict = {
    "__name__": "__main__",
    "__file__": runner_path,
    "__package__": None,
    "__spec__": None,
    "_ADAPTIVE_V4_GIT_OBJECT_LAUNCH_SENTINEL_V1_3_5": {
        "schema_version": 1,
        "launcher": "p2-direct-controller-git-object-launcher-v1-3-5",
        "sealed_runner": True,
        "sealed_inventory": True,
    },
    "_ADAPTIVE_V4_GIT_OBJECT_PYTHON_RUNTIME_V1_3_5": dict(python_runtime),
    "_ADAPTIVE_V4_GIT_OBJECT_SOURCE_PROVENANCE_V1_3_5": source_provenance,
    "_ADAPTIVE_V4_GIT_OBJECT_LAUNCH_ROUTING_V1_3_5": dict(routing),
}
exec(compile(runner_bytes, runner_path, "exec"), globals_dict, globals_dict)
"""


def launch_frozen_runner(
    bundle: Mapping[str, Any],
    runner_bytes: bytes,
    runner_arguments: Sequence[str],
    *,
    entrypoint: str = "matrix",
) -> int:
    repository_root = bundle.get("repository_root")
    _require(isinstance(repository_root, str), "Frozen repository root is invalid.")
    root = _exact_repository_root(Path(cast(str, repository_root)))
    runtime = _python_runtime_binding(root)
    _require(
        bundle.get("python_runtime") == runtime,
        "Frozen source bundle Python runtime binding drifted before launch.",
    )
    encoded_bundle = _canonical_json(dict(bundle))
    _require(
        len(encoded_bundle) <= MAXIMUM_SOURCE_BUNDLE_BYTES,
        "Frozen implementation source bundle exceeds its size limit.",
    )
    encoded_routing = _canonical_json(_launch_routing_binding(bundle, runner_bytes, entrypoint))
    _require(
        len(encoded_routing) <= MAXIMUM_LAUNCH_ROUTING_BYTES,
        "Frozen invocation routing binding exceeds its size limit.",
    )
    runner_fd = _create_sealed_memfd(
        "adaptive-v4-direct-exact-fill-v1-3-5-git-runner", runner_bytes
    )
    bundle_fd: int | None = None
    routing_fd: int | None = None
    try:
        bundle_fd = _create_sealed_memfd(
            "adaptive-v4-direct-exact-fill-v1-3-5-git-source-bundle", encoded_bundle
        )
        routing_fd = _create_sealed_memfd(
            "adaptive-v4-direct-exact-fill-v1-3-5-git-launch-routing", encoded_routing
        )
        environment = _python_environment()
        environment[RUNNER_FD_ENV] = str(runner_fd)
        environment[SOURCE_BUNDLE_FD_ENV] = str(bundle_fd)
        environment[LAUNCH_ROUTING_FD_ENV] = str(routing_fd)
        result = subprocess.run(
            [
                runtime["executable"],
                "-I",
                "-S",
                "-B",
                "-c",
                RUNNER_BOOTSTRAP_SOURCE,
                *runner_arguments,
            ],
            check=False,
            cwd=root,
            env=environment,
            pass_fds=(runner_fd, bundle_fd, routing_fd),
        )
        return result.returncode
    finally:
        if routing_fd is not None:
            os.close(routing_fd)
        if bundle_fd is not None:
            os.close(bundle_fd)
        os.close(runner_fd)


def canonical_command(
    repository_root: Path,
    head_oid: str,
    *,
    entrypoint: str = "matrix",
) -> str:
    root = _exact_repository_root(repository_root)
    _require(_is_object_id(head_oid), "Canonical command HEAD object ID is invalid.")
    _entrypoint_relative_path(entrypoint)
    runtime = _python_runtime_binding(root)
    runtime_digest = _json_digest(runtime)
    git_environment = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in _git_environment().items()
    )
    python_environment = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in _python_environment().items()
    )
    return (
        f"set -euo pipefail; head={head_oid}; "
        f"/usr/bin/env -i {git_environment} "
        f"{shlex.quote(str(PINNED_GIT_EXECUTABLE))} -c core.hooksPath=/dev/null "
        f"-c core.commitGraph=false "
        f"-C {shlex.quote(str(root))} "
        f"cat-file blob $head:{shlex.quote(LAUNCHER_RELATIVE_PATH)} | "
        f"/usr/bin/env -i {python_environment} "
        f"{shlex.quote(cast(str, runtime['executable']))} -I -S -B - "
        f"{shlex.quote(str(root))} $head {runtime_digest} "
        f"{shlex.quote(entrypoint)} --"
    )


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    _require(
        globals().get("__file__") == "<stdin>",
        "Canonical launcher must itself be read from the pinned Git blob over stdin.",
    )
    _require(sys.flags.isolated == 1, "Canonical launcher requires Python isolated mode (-I).")
    _require(sys.flags.no_site == 1, "Canonical launcher requires site suppression (-S).")
    _require(sys.dont_write_bytecode, "Canonical launcher requires bytecode suppression (-B).")
    _require(
        len(arguments) >= 5 and arguments[4] == "--",
        "Usage: <launcher-stdin> REPOSITORY_ROOT PINNED_HEAD PYTHON_RUNTIME_DIGEST "
        "{matrix|audit|summary} -- [entrypoint arguments]",
    )
    root = _exact_repository_root(Path(arguments[0]))
    runtime = _python_runtime_binding(root)
    _require(
        Path(os.path.abspath(sys.executable)) == Path(cast(str, runtime["executable"]))
        and _is_sha256(arguments[2])
        and _json_digest(runtime) == arguments[2],
        "Canonical launcher exact Python runtime differs from its command.",
    )
    pinned_head = pin_starting_head(root, arguments[1])
    entrypoint_path = _entrypoint_relative_path(arguments[3])
    bundle, runner_bytes = build_frozen_source_bundle(
        root,
        pinned_head,
        runner_relative_path=entrypoint_path,
    )
    raise SystemExit(
        launch_frozen_runner(
            bundle,
            runner_bytes,
            arguments[5:],
            entrypoint=arguments[3],
        )
    )


if __name__ == "__main__":
    main()
