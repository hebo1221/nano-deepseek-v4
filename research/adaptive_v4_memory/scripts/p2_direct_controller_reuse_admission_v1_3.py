from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import importlib
import json
import os
import secrets
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import p2_direct_attestation as attestation

# This module is intentionally independent of every v1.3 controller contract and evaluator.
# Admission must be possible before a final v1.3 manifest exists, but it must fail closed until
# the caller supplies that final manifest and a clean result-source checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

QUALITY_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-v1.3"
QUALITY_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-exact-fill-v1-3"
)
ADMISSION_ROOT = QUALITY_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-admission"
DEFAULT_ADMISSION_PATH = ADMISSION_ROOT / "historical-reuse-admission.json"
DEFAULT_GENESIS_PATH = ADMISSION_ROOT / "preheldout-genesis.json"
ADMISSION_STAGING_PREFIX = ".controller-exact-fill-v1-3-admission.staging-"
DIRECT_GPU_SCHEDULER_LOCK_PATH = Path("/tmp/adaptive-v4-direct-gpu0.lock")
QUALITY_MATRIX_SUMMARY_PATH = QUALITY_OUTPUT_ROOT / "controller-matrix.summary.json"
QUALITY_INTEGRITY_OUTPUT_PATH = QUALITY_OUTPUT_ROOT.parent / (
    "controller-exact-fill-v1-3.integrity.json"
)
QUALITY_SUMMARY_OUTPUT_PATH = QUALITY_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3.summary.json"
QUALITY_MATRIX_LOCK_PATH = QUALITY_OUTPUT_ROOT.parent / (
    ".controller-exact-fill-v1-3.p2-direct-controller-matrix-v1-3.lock"
)
QUALITY_WORKER_LEDGER_ROOT = QUALITY_OUTPUT_ROOT.parent / (
    ".controller-exact-fill-v1-3.p2-direct-controller-workers-v1-3"
)
QUALITY_PERSISTENT_SESSION_LEDGER_ROOT = QUALITY_OUTPUT_ROOT.parent / (
    f".{QUALITY_OUTPUT_ROOT.name}.p2-direct-controller-persistent-sessions-v1-3"
)
QUALITY_PERSISTENT_SESSION_LEDGER_LOCK_PATH = (
    QUALITY_PERSISTENT_SESSION_LEDGER_ROOT.parent
    / f"{QUALITY_PERSISTENT_SESSION_LEDGER_ROOT.name}.lock"
)
PROSPECTIVE_QUALITY_PATHS = (
    QUALITY_OUTPUT_ROOT,
    QUALITY_MATRIX_SUMMARY_PATH,
    QUALITY_MATRIX_LOCK_PATH,
    QUALITY_WORKER_LEDGER_ROOT,
    QUALITY_PERSISTENT_SESSION_LEDGER_ROOT,
    QUALITY_PERSISTENT_SESSION_LEDGER_LOCK_PATH,
    QUALITY_INTEGRITY_OUTPUT_PATH,
    QUALITY_SUMMARY_OUTPUT_PATH,
)

HISTORICAL_RESULT_SOURCE_COMMIT = "8c88464d3de39dd98a119ec98cef99a5f7a8c0f5"
HISTORICAL_RESULT_SOURCE_TREE = "8f6a8ced184fee6afb98ceb870167c2ed02a36f4"
HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT = "ce04646b09051e8ad13783eb794f4d92f728fd85"
HISTORICAL_IMPLEMENTATION_SOURCE_TREE = "2d7e26b84d8067360db85c04c9c1e4d4cf35768c"
HISTORICAL_IMPLEMENTATION_DIGEST = (
    "2978634d0c6da3e45d8a97a494f5a319778653ebd7c9a774f44ff72cee238b96"
)
HISTORICAL_MANIFEST_RELATIVE_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-v1-2.json"
)
HISTORICAL_MANIFEST_SHA256 = "eab8e67d9e2d0e162aaf19a637a2978a1a41d71c799570aecfec3c31499754d7"
HISTORICAL_MANIFEST_BYTES = 40_363
HISTORICAL_MANIFEST_EXPERIMENT_ID = "p2-post-rank-direct-controller-v1.2"

V1_1_RESULT_SOURCE_COMMIT = "95339f4dd5b9757c1b513fc6be391fea206b2bc9"
V1_1_RESULT_SOURCE_TREE = "9100d2c4cf00a87a39ffbd1e79f671c3e3b8d6b3"
V1_1_IMPLEMENTATION_SOURCE_COMMIT = "80ef62672ea1f625acd0481e2ae34aa7c4a3f4a3"
V1_1_IMPLEMENTATION_SOURCE_TREE = "4bacc3be6a50b7fc45234121b26c178837f9b5c5"

HISTORICAL_IMPLEMENTATION_PATHS = (
    "pyproject.toml",
    "nano_deepseek_v4",
    "research/adaptive_v4_memory/reports/2026-07-19-p2-direct-training-validator-amendment.md",
    "research/adaptive_v4_memory/reports/2026-07-19-p2-direct-calibration-path-binding-amendment.md",
    "research/adaptive_v4_memory/scripts/adaptive_v4_execution_environment.py",
    "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py",
    "research/adaptive_v4_memory/scripts/freeze_p2_causal_factorial_arms.py",
    "research/adaptive_v4_memory/scripts/p2_direct_attestation.py",
    "research/adaptive_v4_memory/scripts/train_m1_associative_recall.py",
    "research/adaptive_v4_memory/scripts/p2_direct_controller_contract.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_training_matrix.py",
    "research/adaptive_v4_memory/scripts/calibrate_p2_direct_soft_lag.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_calibration_matrix.py",
    "research/adaptive_v4_memory/scripts/validate_p2_direct_top_p_physical_match.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_top_p_physical_matrix.py",
    "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard.py",
    "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix.py",
    "research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity.py",
    "research/adaptive_v4_memory/scripts/summarize_p2_direct_controller.py",
)

HISTORICAL_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct")
HISTORICAL_TRAINING_ROOT = HISTORICAL_ROOT / "training"
HISTORICAL_CALIBRATION_QUARANTINE_ROOT = HISTORICAL_ROOT / "calibration"
HISTORICAL_CALIBRATION_ROOT = HISTORICAL_ROOT / "calibration-v1-2"
HISTORICAL_TOP_P_ROOT = HISTORICAL_ROOT / "top_p_physical_match"
HISTORICAL_TRAINING_LEDGER = HISTORICAL_TRAINING_ROOT / "training-matrix-v1-1.summary.json"
HISTORICAL_CALIBRATION_LEDGER = HISTORICAL_CALIBRATION_ROOT / "calibration-matrix-v1-2.summary.json"
HISTORICAL_TOP_P_LEDGER = HISTORICAL_TOP_P_ROOT / "top-p-physical-matrix.summary.json"

HISTORICAL_TRAINING_LEDGER_SHA256 = (
    "786669b8feb74eef5a4aa1e57dccc3ffada10596a8ae995d78e931daeef06cb5"
)
HISTORICAL_CALIBRATION_LEDGER_SHA256 = (
    "b6da3a7861ac0a7d3d3d94cec9b6031a40270f01ff5eef8a1513761d97cc7b61"
)
HISTORICAL_TOP_P_LEDGER_SHA256 = "ad4b5d2dcdcbc50ccd5008a0927cfb1a3265f904740e22f6ebab23c2b8f40c41"
HISTORICAL_TRAINING_ADMISSION_SHA256 = (
    "34fdabea3bc932e8dd5b2753c4356c62a46492ba849d2468ecef9eaf436ecdf4"
)
HISTORICAL_CALIBRATION_ADMISSION_SHA256 = (
    "a860dfc2aa3a6484b47db167fe77b92970fb7b0c5e9c91301dbca178b2af946b"
)
HISTORICAL_SUPERSEDED_TRAINING_LEDGER_SHA256 = (
    "dc469a9c22ef295ed61022042fd6f1c4ddbd8544adb144986d590fc0f7b2ef7f"
)
HISTORICAL_SUPERSEDED_TRAINING_CLAIM_SHA256 = (
    "60724fc6226380bc8b457cc3e4518c5060beda037fb4878f21b14614bb204309"
)
HISTORICAL_QUARANTINE_LEDGER_SHA256 = (
    "f0dccaa9861e095b297c22a17735b3a379d4f9ed8db628e0bcf222b12da5e426"
)
HISTORICAL_QUARANTINE_CLAIM_SHA256 = (
    "462793153ad22a19223398ef30c2e4624ae247d179a8c29cb250dca1b3bca2cb"
)
HISTORICAL_QUARANTINE_ARTIFACT_SHA256 = (
    "f805d70cb1379cc71c6d6abbe34d579880bd8cec35b72ea816ce9cbc7775d25b"
)

HISTORICAL_ROOT_COUNTS = {
    "training": {"files": 24, "directories": 13},
    "calibration-quarantine": {"files": 3, "directories": 3},
    "calibration-v1-2": {"files": 12, "directories": 13},
    "top-p": {"files": 41, "directories": 33},
}

OLD_CONTROLLER_OUTPUT_ROOT = HISTORICAL_ROOT / "controller"
OLD_CONTROLLER_WORKER_LEDGER_ROOT = HISTORICAL_ROOT / (".controller.p2-direct-controller-workers")
OLD_CONTROLLER_MATRIX_LOCK_PATH = HISTORICAL_ROOT / (".controller.p2-direct-controller-matrix.lock")
CANONICAL_NONOBSERVATION_PATHS = (
    OLD_CONTROLLER_OUTPUT_ROOT,
    OLD_CONTROLLER_WORKER_LEDGER_ROOT,
    OLD_CONTROLLER_MATRIX_LOCK_PATH,
)

TRAINING_SEEDS = (6071406, 6071407, 6071408, 6071409, 6071410)
CALIBRATION_SEEDS = (7071406, 7071407, 7071408, 7071409, 7071410)
EVALUATION_SEEDS = (10071406, 10071407, 10071408, 10071409, 10071410)
SCALES = ("s55", "s151")
BUDGETS = ("2x", "4x")
EXPECTED_QUALITY_SHARDS = 9_000
QUALITY_COORDINATE_DIGEST = "9f7b099051a785a87082d5030e494398098fcb1766574c5df829b2fcd4a21c4a"
FROZEN_EXACT_FILL_ARM_NAMES = (
    "hierarchical-soft-lag+pins",
    "fixed+pins",
    "hierarchical-balanced-fixed+pins",
    "fixed",
    "calibrated-no-pins",
    "calibrated+pins",
    "shuffled-quota",
    "shuffled-quota+pins",
    "local-no-pins",
    "local+pins",
    "hierarchical-soft-lag-no-pins",
    "hierarchical-soft-lag+pins-no-score",
    "hierarchical-soft-lag+pins-no-temporal",
    "hierarchical-soft-lag+pins-no-cross-layer",
    "hierarchical-soft-lag+pins-no-refresh",
    "hierarchical-soft-lag+pins-permuted-quota",
    "hierarchical-soft-lag+pins+fallback",
)

HISTORICAL_RECEIPT_PURPOSE = "p2-direct-v1.3-historical-validation-receipt-v1"
NONOBSERVATION_PURPOSE = "p2-direct-v1.3-canonical-nonobservation-v1"
REUSE_ADMISSION_PURPOSE = "p2-direct-v1.3-reuse-admission-v1"
PREHELDOUT_GENESIS_PURPOSE = "p2-direct-v1.3-preheldout-genesis-v1"
LEGACY_CALIBRATION_PURPOSE = "p2-direct-soft-lag-calibration-v1"
MODULE_ORIGIN_AUDIT_SEMANTICS = (
    "source-only-frozen-inventory-with-sealed-admission-isolated-pycache-and-trusted-venv-v2"
)
ARCHIVED_THIRD_PARTY_RUNTIME_CLAIM = (
    "exact sealed site-packages directory identity with a trusted preinstalled environment; "
    "third-party package bytes and distribution records are not integrity-bound, and same-UID "
    "dependency mutation is outside the admission threat model"
)

RECEIPT_SCHEMA_VERSION = 1
NONOBSERVATION_SCHEMA_VERSION = 1
ADMISSION_SCHEMA_VERSION = 1
GENESIS_SCHEMA_VERSION = 1
SAFE_FILE_MODE = 0o600
SAFE_DIRECTORY_MODE = 0o700

ARCHIVED_MODULE_FD_ENV = "ADAPTIVE_V4_V1_3_ARCHIVED_ADMISSION_MODULE_FD"
ARCHIVED_IMPORT_INVENTORY_FD_ENV = "ADAPTIVE_V4_V1_3_ARCHIVED_IMPORT_INVENTORY_FD"
ARCHIVED_RECEIPT_FD_ENV = "ADAPTIVE_V4_V1_3_ARCHIVED_RECEIPT_FD"
ARCHIVED_PYTHON_RUNTIME_FD_ENV = "ADAPTIVE_V4_V1_3_ARCHIVED_PYTHON_RUNTIME_FD"
ARCHIVED_PYTHON_RELATIVE_PATH = Path(".venv/bin/python")
ARCHIVED_REQUIRED_THIRD_PARTY_MODULES = ("numpy", "safetensors", "torch")
MAXIMUM_PYTHON_EXECUTABLE_BYTES = 128 << 20
ARCHIVED_CHILD_BOOTSTRAP = f"""
import ast
import fcntl
import hashlib
import importlib.machinery
import json
import os
import stat
import sys

p = sys.argv.pop(1)
if not (sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.flags.dont_write_bytecode == 1):
    raise RuntimeError("archived child requires -I -S -B before protected FDs are inherited")
required_seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
runtime_fd = int(os.environ[{ARCHIVED_PYTHON_RUNTIME_FD_ENV!r}])
if fcntl.fcntl(runtime_fd, fcntl.F_GET_SEALS) & required_seals != required_seals:
    raise RuntimeError("sealed archived Python runtime FD is not immutable")
runtime_data = os.pread(runtime_fd, os.fstat(runtime_fd).st_size, 0)
runtime = json.loads(runtime_data.decode("utf-8"))
if runtime_data != json.dumps(
    runtime, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode():
    raise RuntimeError("archived Python runtime binding is not canonical")

def path_within(candidate, prefix):
    try:
        return os.path.commonpath((candidate, prefix)) == prefix
    except ValueError:
        return False

def exact_metadata(path, require_directory=False):
    metadata = os.stat(path, follow_symlinks=False)
    expected_type = stat.S_ISDIR(metadata.st_mode) if require_directory else stat.S_ISREG(metadata.st_mode)
    if not expected_type:
        raise RuntimeError("archived Python runtime object has the wrong type: " + path)
    return {{
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "bytes": metadata.st_size,
    }}

def metadata_identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )

def git_blob_oid(source, expected):
    algorithm = "sha1" if len(expected) == 40 else "sha256"
    digest = hashlib.new(algorithm, usedforsecurity=False)
    digest.update(b"blob " + str(len(source)).encode("ascii") + b"\\0" + source)
    return digest.hexdigest()

def active_python_runtime_binding(repository_root):
    root = os.path.abspath(repository_root)
    if os.path.realpath(root) != root or not os.path.isdir(root):
        raise RuntimeError("archived Python runtime repository root is not exact")
    venv_executable = os.path.abspath(
        os.path.join(root, {ARCHIVED_PYTHON_RELATIVE_PATH.as_posix()!r})
    )
    executable = os.path.realpath("/proc/self/exe")
    if (
        os.path.realpath(sys.executable) != executable
        or os.path.realpath(venv_executable) != executable
        or os.path.abspath(executable) != executable
    ):
        raise RuntimeError("archived Python runtime executable binding drifted")
    metadata = exact_metadata(executable)
    if (
        metadata["uid"] != 0
        or metadata["mode"] & (stat.S_IWGRP | stat.S_IWOTH)
        or not 0 < metadata["bytes"] <= {MAXIMUM_PYTHON_EXECUTABLE_BYTES}
        or not os.access(executable, os.X_OK)
    ):
        raise RuntimeError("archived Python runtime executable metadata is unsafe")
    descriptor = os.open(executable, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        observed_bytes = 0
        while True:
            chunk = os.read(descriptor, 1048576)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > {MAXIMUM_PYTHON_EXECUTABLE_BYTES}:
                raise RuntimeError("archived Python executable exceeds its size limit")
            digest.update(chunk)
        final = os.stat(executable, follow_symlinks=False)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (final.st_dev, final.st_ino, final.st_size)
            or observed_bytes != opened.st_size
        ):
            raise RuntimeError("archived Python executable changed while binding")
    finally:
        os.close(descriptor)
    version_directory = "python{{}}.{{}}".format(sys.version_info.major, sys.version_info.minor)
    site_packages = os.path.abspath(
        os.path.join(root, ".venv", "lib", version_directory, "site-packages")
    )
    if os.path.realpath(site_packages) != site_packages:
        raise RuntimeError("archived repo venv site-packages path is not exact")
    site_metadata = exact_metadata(site_packages, require_directory=True)
    if site_metadata["uid"] != os.getuid() or site_metadata["mode"] & (stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError("archived repo venv site-packages metadata is unsafe")
    return {{
        "schema_version": 1,
        "repository_root": root,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
            sys.version_info.releaselevel,
            sys.version_info.serial,
        ],
        "executable": executable,
        "venv_executable": venv_executable,
        "executable_sha256": digest.hexdigest(),
        "executable_metadata": metadata,
        "base_prefix": os.path.realpath(sys.base_prefix),
        "site_packages": site_packages,
        "site_packages_metadata": site_metadata,
        "required_third_party_modules": list({ARCHIVED_REQUIRED_THIRD_PARTY_MODULES!r}),
    }}

if not isinstance(runtime, dict) or runtime != active_python_runtime_binding(
    runtime.get("repository_root", "")
):
    raise RuntimeError("archived Python runtime binding drifted before protected FD access")
site_packages = runtime["site_packages"]
base_prefix = runtime["base_prefix"]
module_fd = int(os.environ[{ARCHIVED_MODULE_FD_ENV!r}])
inventory_fd = int(os.environ[{ARCHIVED_IMPORT_INVENTORY_FD_ENV!r}])
key_fd = int(os.environ[{attestation.KEY_FD_ENV!r}])
receipt_fd = int(os.environ[{ARCHIVED_RECEIPT_FD_ENV!r}])
if len({{runtime_fd, module_fd, inventory_fd, key_fd, receipt_fd}}) != 5:
    raise RuntimeError("archived child protected file descriptors alias")
if fcntl.fcntl(module_fd, fcntl.F_GET_SEALS) & required_seals != required_seals:
    raise RuntimeError("sealed admission module FD is not immutable")
if fcntl.fcntl(inventory_fd, fcntl.F_GET_SEALS) & required_seals != required_seals:
    raise RuntimeError("sealed archived import inventory FD is not immutable")
module_data = os.pread(module_fd, os.fstat(module_fd).st_size, 0)
inventory_data = os.pread(inventory_fd, os.fstat(inventory_fd).st_size, 0)
inventory = json.loads(inventory_data.decode("utf-8"))
if inventory_data != json.dumps(
    inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode():
    raise RuntimeError("archived import inventory is not canonical")
if set(inventory) != {{"schema_version", "repository_root", "result_source_commit", "files"}}:
    raise RuntimeError("archived import inventory schema drifted")
root = inventory["repository_root"]
if (
    inventory["schema_version"] != 1
    or inventory["result_source_commit"] != {HISTORICAL_RESULT_SOURCE_COMMIT!r}
    or not isinstance(root, str)
    or os.path.realpath(root) != root
    or root != runtime["repository_root"]
    or not isinstance(inventory["files"], list)
):
    raise RuntimeError("archived import inventory header drifted")
scripts = os.path.join(root, "research", "adaptive_v4_memory", "scripts")
allowed = {{}}
sources = {{}}
snapshots = {{}}
root_before = os.stat(root, follow_symlinks=False)
for row in inventory["files"]:
    if set(row) != {{"path", "git_mode", "git_blob_oid", "sha256", "bytes"}}:
        raise RuntimeError("archived import inventory row schema drifted")
    relative = row["path"]
    candidate = os.path.abspath(os.path.join(root, relative))
    if (
        not isinstance(relative, str)
        or os.path.commonpath((candidate, root)) != root
        or candidate in allowed
        or row["git_mode"] not in ("100644", "100755")
        or not isinstance(row["git_blob_oid"], str)
        or len(row["git_blob_oid"]) not in (40, 64)
        or not isinstance(row["sha256"], str)
        or len(row["sha256"]) != 64
        or type(row["bytes"]) is not int
        or row["bytes"] < 0
    ):
        raise RuntimeError("archived import inventory row drifted")
    metadata = os.stat(candidate, follow_symlinks=False)
    mode = stat.S_IMODE(metadata.st_mode)
    executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_nlink != 1
        or bool(mode & stat.S_IWOTH)
        or (row["git_mode"] == "100644" and executable)
        or (row["git_mode"] == "100755" and not executable)
        or os.path.realpath(candidate) != candidate
    ):
        raise RuntimeError("archived import inventory path metadata drifted")
    descriptor = os.open(candidate, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(candidate, follow_symlinks=False)
        if metadata_identity(opened) != metadata_identity(current):
            raise RuntimeError("archived import inventory path changed while opening: " + candidate)
        source = b""
        while True:
            chunk = os.read(descriptor, 1048576)
            if not chunk:
                break
            source += chunk
        final_opened = os.fstat(descriptor)
        final = os.stat(candidate, follow_symlinks=False)
        if not (
            metadata_identity(opened)
            == metadata_identity(final_opened)
            == metadata_identity(final)
        ):
            raise RuntimeError("archived import inventory path changed while reading: " + candidate)
    finally:
        os.close(descriptor)
    if (
        len(source) != row["bytes"]
        or hashlib.sha256(source).hexdigest() != row["sha256"]
        or git_blob_oid(source, row["git_blob_oid"]) != row["git_blob_oid"]
    ):
        raise RuntimeError("archived import inventory path bytes drifted")
    allowed[candidate] = row
    snapshots[candidate] = metadata_identity(opened)
    if candidate.endswith(".py"):
        sources[candidate] = source

if metadata_identity(os.stat(root, follow_symlinks=False)) != metadata_identity(root_before):
    raise RuntimeError("archived import inventory root changed during validation")
for candidate, snapshot in snapshots.items():
    if metadata_identity(os.stat(candidate, follow_symlinks=False)) != snapshot:
        raise RuntimeError("archived import inventory member changed after validation: " + candidate)

frozen_modules = {{}}
script_prefix = "research/adaptive_v4_memory/scripts/"
package_prefix = "nano_deepseek_v4/"
for origin in sorted(sources):
    relative = allowed[origin]["path"]
    if relative.startswith(script_prefix):
        tail = relative[len(script_prefix):]
        if "/" in tail or not tail.endswith(".py") or tail == "__init__.py":
            raise RuntimeError("archived script module path is not canonical: " + relative)
        module_name = tail[:-3]
        is_package = False
    elif relative.startswith(package_prefix):
        tail = relative[:-3]
        parts = tail.split("/")
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        module_name = ".".join(parts)
    else:
        raise RuntimeError("archived Python source is outside protected module roots: " + relative)
    if not module_name or module_name in frozen_modules:
        raise RuntimeError("archived protected module name is empty or duplicated: " + relative)
    frozen_modules[module_name] = (origin, is_package)

stdlib_search_path = []
for entry in sys.path:
    if not entry:
        continue
    candidate = os.path.realpath(entry)
    if path_within(candidate, base_prefix):
        stdlib_search_path.append(entry)
sys.path[:] = stdlib_search_path

def read_frozen_source(origin):
    row = allowed[origin]
    descriptor = os.open(origin, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        current = os.stat(origin, follow_symlinks=False)
        mode = stat.S_IMODE(opened.st_mode)
        executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            or opened.st_uid != current.st_uid
            or opened.st_uid != os.getuid()
            or opened.st_nlink != current.st_nlink
            or opened.st_nlink != 1
            or bool(mode & stat.S_IWOTH)
            or (row["git_mode"] == "100644" and executable)
            or (row["git_mode"] == "100755" and not executable)
        ):
            raise ImportError("frozen repository source metadata changed before execution: " + origin)
        source = b""
        while True:
            chunk = os.read(descriptor, 1048576)
            if not chunk:
                break
            source += chunk
        final = os.stat(origin, follow_symlinks=False)
        final_opened = os.fstat(descriptor)
        if (
            metadata_identity(opened) != snapshots[origin]
            or metadata_identity(opened)
            != metadata_identity(final_opened)
            or metadata_identity(opened) != metadata_identity(final)
            or len(source) != row["bytes"]
            or hashlib.sha256(source).hexdigest() != row["sha256"]
            or git_blob_oid(source, row["git_blob_oid"]) != row["git_blob_oid"]
            or source != sources[origin]
        ):
            raise ImportError("frozen repository source bytes changed before execution: " + origin)
        return sources[origin]
    finally:
        os.close(descriptor)

class FrozenSourceLoader:
    def __init__(self, origin):
        self.origin = origin

    def create_module(self, spec):
        del spec
        return None

    def exec_module(self, module):
        source = read_frozen_source(self.origin)
        module.__file__ = self.origin
        module.__cached__ = None
        exec(compile(source, self.origin, "exec"), module.__dict__, module.__dict__)

def frozen_spec(fullname, origin, is_package):
    spec = importlib.machinery.ModuleSpec(
        fullname,
        FrozenSourceLoader(origin),
        origin=origin,
        is_package=is_package,
    )
    spec.has_location = True
    if is_package:
        spec.submodule_search_locations = [os.path.dirname(origin)]
    return spec

def validate_external_spec(spec):
    if spec is None or spec.origin in ("built-in", "frozen"):
        return spec
    if spec.origin is None:
        locations = spec.submodule_search_locations
        if locations is None or not locations:
            raise ImportError("originless external import rejected before execution")
        for location in locations:
            resolved = os.path.realpath(location)
            if not (path_within(resolved, base_prefix) or path_within(resolved, site_packages)):
                raise ImportError(
                    "external namespace import location rejected before execution: " + resolved
                )
        return spec
    origin = os.path.realpath(os.path.abspath(spec.origin))
    if path_within(origin, site_packages) or path_within(origin, base_prefix):
        return spec
    if path_within(origin, root):
        raise ImportError("non-frozen repository import origin rejected before execution: " + origin)
    raise ImportError("external import origin rejected before execution: " + origin)

class FrozenRepositoryFinder:
    @staticmethod
    def find_spec(fullname, path=None, target=None):
        del target
        protected = frozen_modules.get(fullname)
        if protected is not None:
            return frozen_spec(fullname, protected[0], protected[1])
        return validate_external_spec(importlib.machinery.PathFinder.find_spec(fullname, path))

sys.meta_path = [
    importlib.machinery.BuiltinImporter,
    importlib.machinery.FrozenImporter,
    FrozenRepositoryFinder,
]
sys.path.insert(0, root)
sys.path.insert(0, scripts)
sys.path.append(site_packages)

import_names = set()
for source in sources.values():
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            import_names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            import_names.add(node.module.split(".", 1)[0])
repository_top_levels = {{name.split(".", 1)[0] for name in frozen_modules}}
third_party_names = import_names - set(sys.stdlib_module_names) - repository_top_levels
if third_party_names != set(runtime["required_third_party_modules"]):
    raise RuntimeError("archived required third-party import inventory drifted")
for name in sorted(import_names):
    if name in sys.stdlib_module_names:
        continue
    spec = FrozenRepositoryFinder.find_spec(name)
    if spec is None:
        raise ImportError("archived static import is unavailable: " + name)
    if name in third_party_names:
        if spec.origin is None or not path_within(os.path.realpath(spec.origin), site_packages):
            raise ImportError("archived third-party import escaped repo venv: " + name)

g = {{"__name__": "__main__", "__file__": p, "__package__": None}}
exec(compile(module_data, p, "exec"), g, g)
"""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_oid(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) in {40, 64}
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def canonical_pretty_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _absolute(path: Path, *, repository_root: Path = REPOSITORY_ROOT) -> Path:
    candidate = path if path.is_absolute() else repository_root / path
    return Path(os.path.abspath(candidate))


def _exact_path(
    path: Path,
    *,
    label: str,
    repository_root: Path = REPOSITORY_ROOT,
    must_exist: bool = False,
) -> Path:
    absolute = _absolute(path, repository_root=repository_root)
    _require(not absolute.is_symlink(), f"{label} may not be a symbolic link.")
    try:
        resolved = absolute.resolve(strict=must_exist)
    except OSError as error:
        raise ValueError(f"{label} is missing or cannot be resolved safely.") from error
    _require(resolved == absolute, f"{label} must use its exact resolved path.")
    return absolute


def _runtime_metadata(path: Path, *, require_directory: bool = False) -> dict[str, int]:
    metadata = os.stat(path, follow_symlinks=False)
    expected_type = (
        stat.S_ISDIR(metadata.st_mode) if require_directory else stat.S_ISREG(metadata.st_mode)
    )
    _require(expected_type, f"Archived Python runtime object has the wrong type: {path}")
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "bytes": metadata.st_size,
    }


def _archived_python_runtime_binding(repository_root: Path) -> dict[str, Any]:
    root = _exact_path(repository_root, label="Archived runtime repository root", must_exist=True)
    _require(root.is_dir(), "Archived runtime repository root must be a directory.")
    _require(
        sys.implementation.name == "cpython" and sys.version_info >= (3, 10),
        "Archived validation requires CPython 3.10 or newer.",
    )
    venv_executable = Path(os.path.abspath(root / ARCHIVED_PYTHON_RELATIVE_PATH))
    executable = Path("/proc/self/exe").resolve(strict=True)
    _require(
        Path(sys.executable).resolve(strict=True) == executable
        and venv_executable.resolve(strict=True) == executable,
        "Repository venv Python does not resolve to the active exact interpreter.",
    )
    metadata = _runtime_metadata(executable)
    _require(
        metadata["uid"] == 0
        and metadata["mode"] & (stat.S_IWGRP | stat.S_IWOTH) == 0
        and 0 < metadata["bytes"] <= MAXIMUM_PYTHON_EXECUTABLE_BYTES
        and os.access(executable, os.X_OK),
        "Archived Python runtime executable metadata is unsafe.",
    )
    descriptor = os.open(executable, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        digest = hashlib.sha256()
        observed_bytes = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            observed_bytes += len(chunk)
            _require(
                observed_bytes <= MAXIMUM_PYTHON_EXECUTABLE_BYTES,
                "Archived Python executable exceeds its size limit.",
            )
            digest.update(chunk)
        final = os.stat(executable, follow_symlinks=False)
        _require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (final.st_dev, final.st_ino, final.st_size)
            and observed_bytes == opened.st_size,
            "Archived Python executable changed while binding the runtime.",
        )
    finally:
        os.close(descriptor)
    version_directory = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(
        os.path.abspath(root / ".venv" / "lib" / version_directory / "site-packages")
    )
    _require(
        site_packages.resolve(strict=True) == site_packages,
        "Archived repo venv site-packages path is not exact.",
    )
    site_metadata = _runtime_metadata(site_packages, require_directory=True)
    _require(
        site_metadata["uid"] == os.getuid()
        and site_metadata["mode"] & (stat.S_IWGRP | stat.S_IWOTH) == 0,
        "Archived repo venv site-packages metadata is unsafe.",
    )
    return {
        "schema_version": 1,
        "repository_root": str(root),
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
            sys.version_info.releaselevel,
            sys.version_info.serial,
        ],
        "executable": str(executable),
        "venv_executable": str(venv_executable),
        "executable_sha256": digest.hexdigest(),
        "executable_metadata": metadata,
        "base_prefix": str(Path(sys.base_prefix).resolve(strict=True)),
        "site_packages": str(site_packages),
        "site_packages_metadata": site_metadata,
        "required_third_party_modules": list(ARCHIVED_REQUIRED_THIRD_PARTY_MODULES),
    }


def _archived_third_party_runtime_boundary(runtime: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "site_packages": runtime.get("site_packages"),
        "site_packages_metadata": runtime.get("site_packages_metadata"),
        "required_modules": list(ARCHIVED_REQUIRED_THIRD_PARTY_MODULES),
        "package_bytes_integrity_bound": False,
        "distribution_records_integrity_bound": False,
        "installed_environment_is_trusted_boundary": True,
        "same_uid_dependency_mutation_in_scope": False,
        "claim": ARCHIVED_THIRD_PARTY_RUNTIME_CLAIM,
    }


def _archived_child_environment(
    *,
    runtime_fd: int,
    module_fd: int,
    import_inventory_fd: int,
    key_fd: int,
    receipt_fd: int,
) -> dict[str, str]:
    descriptors = (runtime_fd, module_fd, import_inventory_fd, key_fd, receipt_fd)
    _require(
        all(type(descriptor) is int and descriptor >= 0 for descriptor in descriptors)
        and len(set(descriptors)) == len(descriptors),
        "Archived child protected file descriptors are invalid or aliased.",
    )
    return {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "XDG_CONFIG_HOME": "/nonexistent",
        ARCHIVED_PYTHON_RUNTIME_FD_ENV: str(runtime_fd),
        ARCHIVED_MODULE_FD_ENV: str(module_fd),
        ARCHIVED_IMPORT_INVENTORY_FD_ENV: str(import_inventory_fd),
        ARCHIVED_RECEIPT_FD_ENV: str(receipt_fd),
        attestation.KEY_FD_ENV: str(key_fd),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }


def _archived_child_command(
    *,
    runtime_binding: Mapping[str, Any],
    pycache_prefix: Path,
    module_path: Path,
    repository_root: Path,
    expected_key_id: str,
) -> list[str]:
    executable = runtime_binding.get("executable")
    _require(isinstance(executable, str) and bool(executable), "Archived Python is missing.")
    _require(_is_sha256(expected_key_id), "Archived child expected key ID is invalid.")
    return [
        cast(str, executable),
        "-I",
        "-S",
        "-B",
        "-X",
        f"pycache_prefix={pycache_prefix}",
        "-c",
        ARCHIVED_CHILD_BOOTSTRAP,
        str(module_path),
        "--archived-child",
        "--repository-root",
        str(repository_root),
        "--expected-key-id",
        expected_key_id,
    ]


def _open_secure_regular(path: Path, *, label: str) -> attestation.OpenedRegularFile:
    opened = attestation.open_regular_nofollow(path)
    try:
        metadata = os.fstat(opened.file_descriptor)
        _require(metadata.st_uid == os.getuid(), f"{label} must be owned by the invoking user.")
        _require(metadata.st_nlink == 1, f"{label} must have exactly one hard link.")
        _require(
            stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE,
            f"{label} mode must be 0600.",
        )
        opened.assert_unchanged()
        return opened
    except BaseException:
        opened.close()
        raise


def _read_json_opened(
    opened: attestation.OpenedRegularFile,
    *,
    label: str,
    require_canonical_pretty_bytes: bool,
) -> dict[str, Any]:
    try:
        raw = opened.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON.") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object.")
    if require_canonical_pretty_bytes:
        _require(raw == canonical_pretty_json(payload), f"{label} bytes are not canonical.")
    opened.assert_unchanged()
    return cast(dict[str, Any], payload)


def _load_json_nofollow(
    path: Path,
    *,
    label: str,
    require_canonical_pretty_bytes: bool = False,
) -> tuple[dict[str, Any], attestation.OpenedRegularFile]:
    opened = _open_secure_regular(path, label=label)
    try:
        payload = _read_json_opened(
            opened,
            label=label,
            require_canonical_pretty_bytes=require_canonical_pretty_bytes,
        )
        return payload, opened
    except BaseException:
        opened.close()
        raise


def _digest_bound_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("attestation", None)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = _json_digest(result)
    return result


def _attested_payload(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    purpose: str,
) -> dict[str, Any]:
    result = _digest_bound_payload(payload)
    result["attestation"] = attestation.attest_payload(
        result,
        trust_root=trust_root,
        purpose=purpose,
    )
    return result


def _verify_attested_payload(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    purpose: str,
    label: str,
) -> None:
    payload_sha256 = payload.get("payload_sha256")
    _require(_is_sha256(payload_sha256), f"{label} payload digest is missing or invalid.")
    digest_source = dict(payload)
    envelope = digest_source.pop("attestation", None)
    digest_source.pop("payload_sha256", None)
    _require(
        payload_sha256 == _json_digest(digest_source),
        f"{label} payload digest does not match its contents.",
    )
    _require(isinstance(envelope, Mapping), f"{label} attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation", None)
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=purpose,
    )


def _file_binding(
    path: Path,
    payload: Mapping[str, Any] | None = None,
    *,
    label: str,
) -> dict[str, Any]:
    opened = _open_secure_regular(path, label=label)
    try:
        result: dict[str, Any] = {
            "path": str(opened.path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
        }
        if payload is not None:
            result.update(
                {
                    "experiment_id": payload.get("experiment_id"),
                    "payload_sha256": payload.get("payload_sha256"),
                    "attestation_mac": (
                        payload.get("attestation", {}).get("mac")
                        if isinstance(payload.get("attestation"), Mapping)
                        else None
                    ),
                }
            )
        opened.assert_unchanged()
        return result
    finally:
        opened.close()


def _validate_file_binding(
    binding: Mapping[str, Any],
    *,
    label: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    _require(
        set(binding) >= {"path", "sha256", "bytes"}
        and isinstance(binding.get("path"), str)
        and _is_sha256(binding.get("sha256"))
        and type(binding.get("bytes")) is int
        and cast(int, binding["bytes"]) >= 0,
        f"{label} binding schema is invalid.",
    )
    path = _exact_path(
        Path(cast(str, binding["path"])),
        label=label,
        repository_root=repository_root,
        must_exist=True,
    )
    opened = _open_secure_regular(path, label=label)
    try:
        _require(
            opened.sha256 == binding["sha256"] and opened.bytes == binding["bytes"],
            f"{label} bytes differ from the admitted binding.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    return path


def _validate_admission_bundle_root(root: Path, *, label: str) -> tuple[Path, Path]:
    _require(os.path.lexists(root), f"{label} is missing.")
    root = _exact_path(root, label=label, must_exist=True)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, f"{label} validation requires O_NOFOLLOW support.")
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | cast(int, no_follow),
    )
    opened_root = os.fstat(descriptor)
    current_root = os.stat(root, follow_symlinks=False)
    _require(
        stat.S_ISDIR(opened_root.st_mode)
        and stat.S_ISDIR(current_root.st_mode)
        and (opened_root.st_dev, opened_root.st_ino) == (current_root.st_dev, current_root.st_ino)
        and opened_root.st_uid == current_root.st_uid == os.getuid()
        and stat.S_IMODE(opened_root.st_mode)
        == stat.S_IMODE(current_root.st_mode)
        == SAFE_DIRECTORY_MODE,
        f"{label} ownership, identity, type, or mode is unsafe.",
    )
    expected_names = {DEFAULT_ADMISSION_PATH.name, DEFAULT_GENESIS_PATH.name}
    try:
        names = set(os.listdir(descriptor))
        _require(
            names == expected_names,
            f"{label} must contain exactly admission and genesis.",
        )
        paths: list[Path] = []
        for name in (DEFAULT_ADMISSION_PATH.name, DEFAULT_GENESIS_PATH.name):
            child_descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
                dir_fd=descriptor,
            )
            try:
                child_metadata = os.fstat(child_descriptor)
                current_child = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                _require(
                    stat.S_ISREG(child_metadata.st_mode)
                    and stat.S_ISREG(current_child.st_mode)
                    and (child_metadata.st_dev, child_metadata.st_ino)
                    == (current_child.st_dev, current_child.st_ino)
                    and child_metadata.st_uid == current_child.st_uid == os.getuid()
                    and child_metadata.st_nlink == current_child.st_nlink == 1
                    and stat.S_IMODE(child_metadata.st_mode)
                    == stat.S_IMODE(current_child.st_mode)
                    == SAFE_FILE_MODE,
                    f"{label} file metadata is unsafe: {root / name}",
                )
            finally:
                os.close(child_descriptor)
            paths.append(root / name)
        final_root = os.stat(root, follow_symlinks=False)
        _require(
            (opened_root.st_dev, opened_root.st_ino) == (final_root.st_dev, final_root.st_ino)
            and set(os.listdir(descriptor)) == expected_names,
            f"{label} changed during closed-world validation.",
        )
    finally:
        os.close(descriptor)
    return paths[0], paths[1]


def _validate_final_admission_root(
    *,
    repository_root: Path,
) -> tuple[Path, Path]:
    root = _absolute(ADMISSION_ROOT, repository_root=repository_root)
    return _validate_admission_bundle_root(root, label="Final admission bundle root")


@dataclass(frozen=True)
class QualityContext:
    manifest_path: Path
    manifest_binding: dict[str, Any]
    source: dict[str, str | bool]
    implementation_paths: tuple[str, ...]
    repository_root: Path


@dataclass(frozen=True)
class AdmittedCheckpoint:
    path: Path
    public_binding: dict[str, Any]


@dataclass(frozen=True)
class AdmittedCalibration:
    path: Path
    public_binding: dict[str, Any]
    checkpoint_binding: dict[str, Any]


@dataclass(frozen=True)
class ValidatedReuseAdmission:
    payload: dict[str, Any]
    public_binding: dict[str, Any]
    calibrations: dict[tuple[str, int], AdmittedCalibration]
    checkpoints: dict[tuple[str, int], AdmittedCheckpoint]
    quality_context: QualityContext
    execution_environment_projection: dict[str, Any]


@dataclass(frozen=True)
class ValidatedPreheldoutGenesis:
    payload: dict[str, Any]
    public_binding: dict[str, Any]


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    check: bool = True,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repository_root), *arguments],
        check=check,
        capture_output=True,
        text=text,
        env=_git_environment(),
    )


def _source_state(repository_root: Path) -> dict[str, str | bool]:
    commit = _git(repository_root, ["rev-parse", "HEAD"]).stdout.strip()
    _require(_is_git_oid(commit), "Quality source commit is invalid.")
    dirty = bool(
        _git(repository_root, ["status", "--porcelain=v1", "--untracked-files=all"]).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def _implementation_index_digest(paths: tuple[str, ...], entries: tuple[str, ...]) -> str:
    return _json_digest(
        {
            "schema_version": 1,
            "implementation_paths": list(paths),
            "git_index_entries": list(entries),
        }
    )


def _parse_index_entries(raw: bytes, *, label: str) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for entry in (item for item in raw.split(b"\0") if item):
        metadata, separator, raw_path = entry.partition(b"\t")
        fields = metadata.decode("ascii").split()
        _require(separator == b"\t" and len(fields) == 3, f"{label} entry is malformed.")
        mode, object_id, stage = fields
        path = raw_path.decode("utf-8")
        _require(
            mode in {"100644", "100755"} and _is_git_oid(object_id) and stage == "0" and bool(path),
            f"{label} contains a symlink, submodule, non-blob, or non-stage-zero entry.",
        )
        parsed.append((path, f"{mode} {object_id} 0\t{path}"))
    _require(len(parsed) == len({path for path, _entry in parsed}), f"{label} repeats a path.")
    canonical = tuple(sorted(parsed))
    _require(tuple(parsed) == canonical, f"{label} ordering is not canonical.")
    return canonical


def _parse_nul_relative_paths(raw: bytes, *, label: str) -> tuple[str, ...]:
    if not raw:
        return ()
    _require(raw.endswith(b"\0"), f"{label} output is not NUL-terminated.")
    encoded_paths = raw[:-1].split(b"\0")
    _require(all(encoded_paths), f"{label} output contains an empty path.")
    paths = tuple(encoded_path.decode("utf-8", errors="strict") for encoded_path in encoded_paths)
    _require(
        all(
            path
            and not path.startswith("/")
            and all(component not in {"", ".", ".."} for component in path.split("/"))
            for path in paths
        ),
        f"{label} output contains a non-canonical or unsafe relative path.",
    )
    _require(len(paths) == len(set(paths)), f"{label} output repeats a path.")
    return tuple(sorted(paths))


def _index_inventory(repository_root: Path, paths: tuple[str, ...]) -> tuple[str, ...]:
    raw = _git(
        repository_root,
        ["ls-files", "-s", "-z", "--", *paths],
        text=False,
    ).stdout
    parsed = _parse_index_entries(raw, label="Implementation index")
    raw_untracked = _git(
        repository_root,
        ["ls-files", "--others", "--exclude-standard", "-z", "--", *paths],
        text=False,
    ).stdout
    untracked = _parse_nul_relative_paths(
        raw_untracked,
        label="Implementation untracked-file inventory",
    )
    _require(
        not untracked,
        "Untracked files exist inside the implementation inventory: "
        + canonical_json(list(untracked)).decode("ascii"),
    )
    return tuple(entry for _path, entry in parsed)


def _commit_inventory(
    repository_root: Path,
    commit: str,
    paths: tuple[str, ...],
) -> tuple[tuple[str, str, str], ...]:
    _require(_is_git_oid(commit), "Historical commit ID is invalid.")
    commit_check = _git(repository_root, ["cat-file", "-e", f"{commit}^{{commit}}"], check=False)
    _require(commit_check.returncode == 0, "Historical commit object is unavailable.")
    raw = _git(
        repository_root,
        ["ls-tree", "-r", "-z", "--full-tree", commit, "--", *paths],
        text=False,
    ).stdout
    parsed: list[tuple[str, str, str]] = []
    for entry in (item for item in raw.split(b"\0") if item):
        metadata, separator, raw_path = entry.partition(b"\t")
        fields = metadata.decode("ascii").split()
        _require(separator == b"\t" and len(fields) == 3, "Commit-tree entry is malformed.")
        mode, object_type, object_id = fields
        path = raw_path.decode("utf-8")
        _require(
            mode in {"100644", "100755"}
            and object_type == "blob"
            and _is_git_oid(object_id)
            and bool(path),
            "Commit tree contains a symlink, submodule, or non-blob entry.",
        )
        parsed.append((path, mode, object_id))
    _require(
        len(parsed) == len({path for path, _mode, _oid in parsed}), "Commit tree repeats a path."
    )
    canonical = tuple(sorted(parsed))
    _require(tuple(parsed) == canonical, "Commit-tree ordering is not canonical.")
    return canonical


def _commit_inventory_digest(
    inventory: Sequence[tuple[str, str, str]],
    paths: tuple[str, ...],
) -> str:
    entries = tuple(f"{mode} {object_id} 0\t{path}" for path, mode, object_id in inventory)
    return _implementation_index_digest(paths, entries)


def _assert_tree_object(repository_root: Path, commit: str, expected_tree: str) -> None:
    observed = _git(repository_root, ["rev-parse", f"{commit}^{{tree}}"]).stdout.strip()
    _require(observed == expected_tree, f"Commit {commit} tree object drifted.")


def _assert_ancestor(repository_root: Path, ancestor: str, descendant: str) -> None:
    result = _git(
        repository_root,
        ["merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
    )
    _require(result.returncode == 0, f"{ancestor} is not an ancestor of {descendant}.")


def _assert_tracked_source_metadata(metadata: os.stat_result, git_mode: str, path: str) -> None:
    live_mode = stat.S_IMODE(metadata.st_mode)
    executable = bool(live_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    _require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and metadata.st_nlink == 1
        and not bool(live_mode & stat.S_IWOTH)
        and ((git_mode == "100755" and executable) or (git_mode == "100644" and not executable)),
        f"Historical runtime path metadata drifted: {path}",
    )


def _git_blob_batch(repository_root: Path, object_ids: Sequence[str]) -> dict[str, bytes]:
    unique = tuple(dict.fromkeys(object_ids))
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository_root),
            "cat-file",
            "--batch",
        ],
        input="".join(f"{object_id}\n" for object_id in unique).encode("ascii"),
        check=True,
        capture_output=True,
        env=_git_environment(),
    )
    raw = completed.stdout
    cursor = 0
    result: dict[str, bytes] = {}
    for requested in unique:
        end = raw.find(b"\n", cursor)
        _require(end >= 0, "Git batch blob header is truncated.")
        fields = raw[cursor:end].decode("ascii").split()
        _require(
            len(fields) == 3
            and fields[0] == requested
            and fields[1] == "blob"
            and fields[2].isdigit(),
            "Git batch object is not the requested blob.",
        )
        size = int(fields[2])
        start = end + 1
        finish = start + size
        _require(
            finish < len(raw) and raw[finish : finish + 1] == b"\n",
            "Git batch blob payload is truncated.",
        )
        result[requested] = raw[start:finish]
        cursor = finish + 1
    _require(cursor == len(raw), "Git batch blob response contains trailing bytes.")
    return result


def _quality_live_implementation_inventory(
    repository_root: Path,
    *,
    paths: tuple[str, ...],
    source_commit: str,
    expected_digest: str,
) -> dict[str, Any]:
    raw_index = _git(
        repository_root,
        ["ls-files", "-s", "-z", "--", *paths],
        text=False,
    ).stdout
    parsed_index = _parse_index_entries(raw_index, label="Quality implementation index")
    index_entries = tuple(entry for _path, entry in parsed_index)
    _require(
        _implementation_index_digest(paths, index_entries) == expected_digest,
        "Quality stage-zero implementation digest drifted.",
    )
    frozen = _commit_inventory(repository_root, source_commit, paths)
    _require(
        _commit_inventory_digest(frozen, paths) == expected_digest,
        "Quality frozen implementation commit digest drifted.",
    )
    expected_entries = tuple(f"{mode} {object_id} 0\t{path}" for path, mode, object_id in frozen)
    _require(
        index_entries == expected_entries,
        "Quality stage-zero blobs differ from the frozen implementation commit.",
    )
    untracked = _git(
        repository_root,
        ["ls-files", "--others", "--exclude-standard", "-z", "--", *paths],
        text=False,
    ).stdout
    _require(not untracked, "Untracked files exist inside the quality implementation inventory.")
    blobs = _git_blob_batch(repository_root, [object_id for _path, _mode, object_id in frozen])
    rows: list[dict[str, Any]] = []
    for path, git_mode, object_id in frozen:
        candidate = repository_root / path
        _require(
            not candidate.is_symlink() and candidate.resolve(strict=True) == candidate,
            f"Quality implementation path is not exact: {path}",
        )
        opened = attestation.open_regular_nofollow(candidate)
        try:
            _assert_tracked_source_metadata(os.fstat(opened.file_descriptor), git_mode, path)
            source = opened.read_bytes()
            _require(
                source == blobs[object_id],
                f"Quality live implementation bytes differ from stage zero and freeze: {path}",
            )
            rows.append(
                {
                    "path": path,
                    "git_mode": git_mode,
                    "git_blob_oid": object_id,
                    "sha256": hashlib.sha256(source).hexdigest(),
                    "bytes": len(source),
                }
            )
            opened.assert_unchanged()
        finally:
            opened.close()
    return {
        "digest": _json_digest(rows),
        "file_count": len(rows),
        "rows": tuple(rows),
    }


def _assert_live_inventory_matches_historical(repository_root: Path) -> dict[str, Any]:
    historical = _commit_inventory(
        repository_root,
        HISTORICAL_RESULT_SOURCE_COMMIT,
        HISTORICAL_IMPLEMENTATION_PATHS,
    )
    historical_digest = _commit_inventory_digest(historical, HISTORICAL_IMPLEMENTATION_PATHS)
    _require(
        historical_digest == HISTORICAL_IMPLEMENTATION_DIGEST,
        "Historical result-source implementation digest drifted.",
    )
    implementation = _commit_inventory(
        repository_root,
        HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
        HISTORICAL_IMPLEMENTATION_PATHS,
    )
    _require(
        _commit_inventory_digest(implementation, HISTORICAL_IMPLEMENTATION_PATHS)
        == HISTORICAL_IMPLEMENTATION_DIGEST,
        "Historical implementation-source digest drifted.",
    )
    _require(historical == implementation, "Result-source implementation bytes differ from freeze.")

    live_entries = _index_inventory(repository_root, HISTORICAL_IMPLEMENTATION_PATHS)
    expected_entries = tuple(
        f"{mode} {object_id} 0\t{path}" for path, mode, object_id in historical
    )
    _require(live_entries == expected_entries, "Live canonical v1.2 inventory differs from 8c884.")

    byte_rows: list[dict[str, Any]] = []
    for path, mode, object_id in historical:
        candidate = repository_root / path
        opened = attestation.open_regular_nofollow(candidate)
        try:
            metadata = os.fstat(opened.file_descriptor)
            _assert_tracked_source_metadata(metadata, mode, path)
            blob = _git(
                repository_root,
                ["cat-file", "blob", object_id],
                text=False,
            ).stdout
            live = opened.read_bytes()
            _require(live == blob, f"Historical runtime bytes drifted: {path}")
            byte_rows.append(
                {
                    "path": path,
                    "mode": mode,
                    "git_blob_oid": object_id,
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                }
            )
            opened.assert_unchanged()
        finally:
            opened.close()
    return {
        "implementation_paths": list(HISTORICAL_IMPLEMENTATION_PATHS),
        "git_index_digest": historical_digest,
        "tree_byte_inventory_digest": _json_digest(byte_rows),
        "tree_byte_inventory_count": len(byte_rows),
    }


def _historical_import_inventory_payload(repository_root: Path) -> dict[str, Any]:
    inventory = _commit_inventory(
        repository_root,
        HISTORICAL_RESULT_SOURCE_COMMIT,
        HISTORICAL_IMPLEMENTATION_PATHS,
    )
    blobs = _git_blob_batch(repository_root, [object_id for _path, _mode, object_id in inventory])
    rows: list[dict[str, Any]] = []
    for path, git_mode, object_id in inventory:
        candidate = repository_root / path
        opened = attestation.open_regular_nofollow(candidate)
        try:
            _assert_tracked_source_metadata(os.fstat(opened.file_descriptor), git_mode, path)
            source = opened.read_bytes()
            _require(
                source == blobs[object_id],
                f"Historical import source differs from its frozen blob: {path}",
            )
            rows.append(
                {
                    "path": path,
                    "git_mode": git_mode,
                    "git_blob_oid": object_id,
                    "sha256": hashlib.sha256(source).hexdigest(),
                    "bytes": len(source),
                }
            )
            opened.assert_unchanged()
        finally:
            opened.close()
    return {
        "schema_version": 1,
        "repository_root": str(repository_root),
        "result_source_commit": HISTORICAL_RESULT_SOURCE_COMMIT,
        "files": rows,
    }


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sealed_admission_module_binding() -> dict[str, Any]:
    raw_descriptor = os.environ.get(ARCHIVED_MODULE_FD_ENV)
    _require(
        raw_descriptor is not None and raw_descriptor.isdigit(),
        "Archived admission sealed-module descriptor is missing.",
    )
    descriptor = int(cast(str, raw_descriptor))
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    _require(
        fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
        "Archived admission module descriptor is not fully sealed.",
    )
    raw = os.pread(descriptor, os.fstat(descriptor).st_size, 0)
    _require(bool(raw), "Archived admission sealed-module snapshot is empty.")
    return {
        "path": str(Path(__file__).resolve(strict=True)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "transport": "sealed-memfd-exec",
    }


def _sealed_archived_python_runtime_binding() -> dict[str, Any]:
    raw_descriptor = os.environ.get(ARCHIVED_PYTHON_RUNTIME_FD_ENV)
    _require(
        raw_descriptor is not None and raw_descriptor.isdigit(),
        "Archived Python sealed-runtime descriptor is missing.",
    )
    descriptor = int(cast(str, raw_descriptor))
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    _require(
        fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
        "Archived Python runtime descriptor is not fully sealed.",
    )
    raw = attestation.read_all_fd(descriptor, maximum_bytes=1 << 20)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Archived Python sealed-runtime binding is not valid JSON.") from error
    _require(
        isinstance(payload, dict) and raw == canonical_json(payload),
        "Archived Python sealed-runtime binding is not canonical.",
    )
    repository_root = payload.get("repository_root")
    _require(
        isinstance(repository_root, str)
        and payload == _archived_python_runtime_binding(Path(repository_root)),
        "Archived Python sealed-runtime binding drifted before origin audit.",
    )
    return cast(dict[str, Any], payload)


def _audit_loaded_repo_module_origins(
    *,
    repository_root: Path,
    detached_root: Path,
    allowed_inventory: Sequence[tuple[str, str, str]],
    sealed_admission: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {path: (mode, object_id) for path, mode, object_id in allowed_inventory}
    sealed_runtime = _sealed_archived_python_runtime_binding()
    sealed_site_packages = Path(cast(str, sealed_runtime["site_packages"]))
    _require(
        sealed_site_packages.resolve(strict=True) == sealed_site_packages,
        "Archived Python sealed site-packages path is not exact.",
    )
    environment_prefixes = tuple(
        dict.fromkeys(
            Path(value).resolve(strict=True)
            for value in (sys.prefix, sys.base_prefix)
            if value and Path(value).exists()
        )
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for module_name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        raw_origins: list[object] = [getattr(module, "__file__", None)]
        specification = getattr(module, "__spec__", None)
        raw_origins.append(getattr(specification, "origin", None))
        for raw_origin in raw_origins:
            if not isinstance(raw_origin, str) or raw_origin in {"built-in", "frozen"}:
                continue
            if not os.path.isabs(raw_origin):
                raw_primary = getattr(module, "__file__", None)
                top_level_name = module_name.split(".", 1)[0]
                top_level_module = sys.modules.get(top_level_name)
                raw_top_level_primary = getattr(top_level_module, "__file__", None)
                top_level_is_sealed_third_party = (
                    top_level_name in ARCHIVED_REQUIRED_THIRD_PARTY_MODULES
                    and isinstance(raw_top_level_primary, str)
                    and os.path.isabs(raw_top_level_primary)
                    and _path_is_within(
                        Path(raw_top_level_primary).resolve(strict=True), sealed_site_packages
                    )
                )
                if top_level_is_sealed_third_party:
                    continue
                _require(
                    isinstance(raw_primary, str) and os.path.isabs(raw_primary),
                    f"Archived child loaded module has an unanchored relative origin: {module_name}",
                )
                primary = Path(cast(str, raw_primary)).resolve(strict=True)
                _require(
                    _path_is_within(primary, sealed_site_packages)
                    or any(_path_is_within(primary, prefix) for prefix in environment_prefixes),
                    f"Archived child loaded relative origin outside its sealed runtime: {module_name}",
                )
                continue
            origin = Path(os.path.abspath(raw_origin))
            try:
                resolved_origin = origin.resolve(strict=True)
            except OSError as error:
                raise ValueError(
                    f"Archived child loaded module origin is missing: {origin}"
                ) from error
            key = (module_name, str(origin))
            if key in seen:
                continue
            seen.add(key)
            if _path_is_within(resolved_origin, sealed_site_packages):
                _require(
                    _path_is_within(origin, sealed_site_packages),
                    f"Archived third-party module origin escaped its sealed site-packages path: {origin}",
                )
                continue
            if any(_path_is_within(resolved_origin, prefix) for prefix in environment_prefixes):
                continue
            source_location: str | None = None
            relative: Path | None = None
            if _path_is_within(origin, repository_root):
                source_location = "canonical-live"
                relative = origin.relative_to(repository_root)
            elif _path_is_within(origin, detached_root):
                source_location = "detached-result-source"
                relative = origin.relative_to(detached_root)
            if relative is None or source_location is None:
                continue
            relative_name = relative.as_posix()
            _require(
                origin.suffix == ".py" and relative_name in allowed,
                f"Archived child imported a non-frozen repository module origin: {origin}",
            )
            _require(
                not origin.is_symlink() and origin.resolve(strict=True) == origin,
                f"Archived child module origin is not an exact regular source path: {origin}",
            )
            git_mode, object_id = allowed[relative_name]
            opened = attestation.open_regular_nofollow(origin)
            try:
                _assert_tracked_source_metadata(
                    os.fstat(opened.file_descriptor), git_mode, relative_name
                )
                blob = _git(
                    repository_root,
                    ["cat-file", "blob", object_id],
                    text=False,
                ).stdout
                source = opened.read_bytes()
                _require(
                    source == blob,
                    f"Archived child loaded module bytes outside the frozen blob: {origin}",
                )
                rows.append(
                    {
                        "module": module_name,
                        "origin": f"{source_location}:{relative_name}",
                        "relative_path": relative_name,
                        "source_location": source_location,
                        "git_mode": git_mode,
                        "git_blob_oid": object_id,
                        "sha256": hashlib.sha256(source).hexdigest(),
                        "bytes": len(source),
                    }
                )
                opened.assert_unchanged()
            finally:
                opened.close()
    rows.sort(key=lambda row: (cast(str, row["module"]), cast(str, row["origin"])))
    _require(bool(rows), "Archived child module-origin audit found no frozen source modules.")
    _require(
        set(sealed_admission) == {"path", "sha256", "bytes", "transport"}
        and _is_sha256(sealed_admission.get("sha256"))
        and type(sealed_admission.get("bytes")) is int
        and cast(int, sealed_admission["bytes"]) > 0
        and sealed_admission.get("transport") == "sealed-memfd-exec",
        "Archived admission sealed-module binding is invalid.",
    )
    third_party_runtime_boundary = _archived_third_party_runtime_boundary(sealed_runtime)
    return {
        "semantics": MODULE_ORIGIN_AUDIT_SEMANTICS,
        "count": len(rows) + 1,
        "digest": _json_digest(
            {
                "origins": rows,
                "sealed_admission": dict(sealed_admission),
                "third_party_runtime_boundary": third_party_runtime_boundary,
            }
        ),
        "origins": rows,
        "sealed_admission": dict(sealed_admission),
        "third_party_runtime_boundary": third_party_runtime_boundary,
    }


def establish_quality_context(
    manifest_path: Path,
    *,
    experiment_id: str = QUALITY_EXPERIMENT_ID,
    implementation_paths: Sequence[str],
    repository_root: Path = REPOSITORY_ROOT,
) -> QualityContext:
    root = _exact_path(repository_root, label="Repository root", must_exist=True)
    _require(root.is_dir(), "Repository root is not a directory.")
    source = _source_state(root)
    _require(source["dirty"] is False, "Quality context requires a clean result-source checkout.")
    path = _exact_path(
        manifest_path,
        label="Final v1.3 manifest",
        repository_root=root,
        must_exist=True,
    )
    payload, opened = _load_json_nofollow(
        path,
        label="Final v1.3 manifest",
        require_canonical_pretty_bytes=True,
    )
    try:
        contract = importlib.import_module("p2_direct_controller_contract_v1_3")
        raw_validator = getattr(contract, "validate_manifest_payload", None)
        canonical_paths = getattr(contract, "IMPLEMENTATION_PATHS", None)
        canonical_experiment_id = getattr(contract, "EXPERIMENT_ID", None)
        _require(callable(raw_validator), "Canonical v1.3 manifest validator is unavailable.")
        validator = cast(Callable[..., Any], raw_validator)
        _require(
            canonical_experiment_id == experiment_id
            and tuple(canonical_paths or ()) == tuple(implementation_paths),
            "Caller manifest boundary differs from the canonical v1.3 contract.",
        )
        validated_manifest = validator(dict(payload), verify_implementation=True)
        _require(
            validated_manifest == payload,
            "Canonical v1.3 validator changed or rejected the manifest payload.",
        )
        _require(payload.get("experiment_id") == experiment_id, "Wrong v1.3 experiment manifest.")
        implementation = payload.get("implementation")
        _require(isinstance(implementation, Mapping), "Manifest implementation binding is missing.")
        paths = tuple(implementation_paths)
        _require(
            tuple(cast(Mapping[str, Any], implementation).get("paths", ())) == paths,
            "Manifest implementation inventory differs from the caller's exact inventory.",
        )
        source_commit = cast(Mapping[str, Any], implementation).get("source_commit")
        tree_digest = cast(Mapping[str, Any], implementation).get("tree_digest")
        _require(_is_git_oid(source_commit), "Manifest implementation commit is invalid.")
        _require(_is_sha256(tree_digest), "Manifest implementation digest is invalid.")
        current_entries = _index_inventory(root, paths)
        _require(
            _implementation_index_digest(paths, current_entries) == tree_digest,
            "Current implementation inventory differs from the final manifest.",
        )
        frozen_inventory = _commit_inventory(root, cast(str, source_commit), paths)
        _require(
            _commit_inventory_digest(frozen_inventory, paths) == tree_digest,
            "Manifest implementation commit does not reproduce its tree digest.",
        )
        live_inventory = _quality_live_implementation_inventory(
            root,
            paths=paths,
            source_commit=cast(str, source_commit),
            expected_digest=cast(str, tree_digest),
        )
        _assert_ancestor(root, cast(str, source_commit), cast(str, source["commit"]))
        public_attestation = payload.get("attestation")
        _require(
            isinstance(public_attestation, Mapping), "Manifest attestation contract is missing."
        )
        expected_contract = attestation.public_manifest_contract(
            str(cast(Mapping[str, Any], public_attestation).get("key_id", ""))
        )
        _require(
            dict(cast(Mapping[str, Any], public_attestation)) == expected_contract,
            "Manifest attestation contract drifted.",
        )
        binding = {
            "path": str(path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
            "experiment_id": experiment_id,
            "implementation_source_commit": source_commit,
            "implementation_digest": tree_digest,
            "live_implementation_inventory_digest": live_inventory["digest"],
            "live_implementation_file_count": live_inventory["file_count"],
            "attestation": expected_contract,
        }
        opened.assert_unchanged()
    finally:
        opened.close()
    context = QualityContext(
        manifest_path=path,
        manifest_binding=binding,
        source={"commit": cast(str, source["commit"]), "dirty": False},
        implementation_paths=paths,
        repository_root=root,
    )
    assert_quality_context_unchanged(context)
    return context


def assert_quality_context_unchanged(context: QualityContext) -> None:
    _require(type(context) is QualityContext, "Quality context must be canonical.")
    current = _source_state(context.repository_root)
    _require(current == context.source, "Quality result-source commit or cleanliness changed.")
    opened = _open_secure_regular(context.manifest_path, label="Final v1.3 manifest")
    try:
        _require(
            opened.sha256 == context.manifest_binding["sha256"]
            and opened.bytes == context.manifest_binding["bytes"],
            "Final v1.3 manifest bytes changed.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    live_inventory = _quality_live_implementation_inventory(
        context.repository_root,
        paths=context.implementation_paths,
        source_commit=cast(str, context.manifest_binding["implementation_source_commit"]),
        expected_digest=cast(str, context.manifest_binding["implementation_digest"]),
    )
    _require(
        live_inventory["digest"] == context.manifest_binding["live_implementation_inventory_digest"]
        and live_inventory["file_count"]
        == context.manifest_binding["live_implementation_file_count"],
        "Quality live implementation inventory changed.",
    )


def quality_implementation_inventory(context: QualityContext) -> tuple[dict[str, Any], ...]:
    """Return exact live rows only after stage-zero/frozen-commit/source-byte replay."""

    _require(type(context) is QualityContext, "Quality context must be canonical.")
    _require(
        _source_state(context.repository_root) == context.source,
        "Quality result-source commit or cleanliness changed.",
    )
    opened = _open_secure_regular(context.manifest_path, label="Final v1.3 manifest")
    try:
        _require(
            opened.sha256 == context.manifest_binding["sha256"]
            and opened.bytes == context.manifest_binding["bytes"],
            "Final v1.3 manifest bytes changed.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    live = _quality_live_implementation_inventory(
        context.repository_root,
        paths=context.implementation_paths,
        source_commit=cast(str, context.manifest_binding["implementation_source_commit"]),
        expected_digest=cast(str, context.manifest_binding["implementation_digest"]),
    )
    _require(
        live["digest"] == context.manifest_binding["live_implementation_inventory_digest"]
        and live["file_count"] == context.manifest_binding["live_implementation_file_count"],
        "Quality live implementation inventory changed.",
    )
    return tuple(dict(row) for row in cast(tuple[dict[str, Any], ...], live["rows"]))


def _scan_closed_world_root(
    root: Path,
    *,
    label: str,
    expected_files: int,
    expected_directories: int,
) -> dict[str, Any]:
    absolute = _exact_path(root, label=label, must_exist=True)
    _require(absolute.is_dir(), f"{label} must be a directory.")
    rows: list[dict[str, Any]] = []
    file_count = 0
    directory_count = 0
    stack = [absolute]
    while stack:
        directory = stack.pop()
        metadata = os.stat(directory, follow_symlinks=False)
        _require(
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and stat.S_IMODE(metadata.st_mode) == SAFE_DIRECTORY_MODE,
            f"{label} contains an unsafe directory.",
        )
        relative_directory = directory.relative_to(absolute).as_posix()
        rows.append(
            {
                "kind": "directory",
                "path": "." if relative_directory == "." else relative_directory,
                "mode": "0700",
            }
        )
        directory_count += 1
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name, reverse=True)
        for child in children:
            child_path = Path(child.path)
            child_metadata = child.stat(follow_symlinks=False)
            _require(not stat.S_ISLNK(child_metadata.st_mode), f"{label} contains a symlink.")
            if stat.S_ISDIR(child_metadata.st_mode):
                stack.append(child_path)
                continue
            _require(
                stat.S_ISREG(child_metadata.st_mode)
                and child_metadata.st_uid == os.getuid()
                and child_metadata.st_nlink == 1
                and stat.S_IMODE(child_metadata.st_mode) == SAFE_FILE_MODE,
                f"{label} contains an unsafe or non-regular file.",
            )
            opened = attestation.open_regular_nofollow(child_path)
            try:
                rows.append(
                    {
                        "kind": "file",
                        "path": child_path.relative_to(absolute).as_posix(),
                        "mode": "0600",
                        "bytes": opened.bytes,
                        "sha256": opened.sha256,
                    }
                )
                opened.assert_unchanged()
            finally:
                opened.close()
            file_count += 1
    rows.sort(key=lambda item: (cast(str, item["path"]), cast(str, item["kind"])))
    _require(
        file_count == expected_files and directory_count == expected_directories,
        f"{label} closed-world count drifted.",
    )
    return {
        "path": str(absolute),
        "files": file_count,
        "directories": directory_count,
        "inventory_digest": _json_digest(rows),
    }


def _assert_canonical_nonobservation_paths_absent(
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    paths = tuple(
        _absolute(path, repository_root=repository_root) for path in CANONICAL_NONOBSERVATION_PATHS
    )
    for path in paths:
        _require(
            not os.path.lexists(path),
            f"Canonical v1.2 held-out path exists and invalidates non-observation: {path}",
        )
    return tuple(str(path) for path in paths)


def _assert_prospective_quality_paths_absent(
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    paths = tuple(
        _absolute(path, repository_root=repository_root) for path in PROSPECTIVE_QUALITY_PATHS
    )
    for path in paths:
        _require(
            not os.path.lexists(path),
            f"Prospective v1.3 quality path exists before admission: {path}",
        )
    return tuple(str(path) for path in paths)


def build_canonical_nonobservation(
    *,
    trust_root: attestation.TrustRoot,
    quality_context: QualityContext,
    historical_receipt: Mapping[str, Any],
    admission_nonce: str,
) -> dict[str, Any]:
    _require(type(quality_context) is QualityContext, "Quality context must be canonical.")
    assert_quality_context_unchanged(quality_context)
    _verify_historical_receipt(historical_receipt, trust_root=trust_root)
    _require(_is_sha256(admission_nonce), "Admission nonce must contain 256 random bits.")
    absent = _assert_canonical_nonobservation_paths_absent(
        repository_root=quality_context.repository_root
    )
    prospective_absent = _assert_prospective_quality_paths_absent(
        repository_root=quality_context.repository_root
    )
    payload = {
        "schema_version": NONOBSERVATION_SCHEMA_VERSION,
        "artifact_type": "direct-controller-canonical-nonobservation",
        "experiment_id": quality_context.manifest_binding["experiment_id"],
        "status": "terminal",
        "admission_nonce": admission_nonce,
        "quality_source": quality_context.source,
        "quality_manifest": quality_context.manifest_binding,
        "historical_receipt_payload_sha256": historical_receipt["payload_sha256"],
        "historical_result_source_commit": HISTORICAL_RESULT_SOURCE_COMMIT,
        "evaluation_seed_namespace": list(EVALUATION_SEEDS),
        "evaluation_seed_namespace_reselected": False,
        "canonical_absent_paths": list(absent),
        "prospective_quality_absent_paths": list(prospective_absent),
        "canonical_quality_output_count": 0,
        "canonical_worker_ledger_count": 0,
        "canonical_quality_matrix_lock_count": 0,
        "canonical_integrity_output_count": 0,
        "canonical_summary_output_count": 0,
        "evaluation_inputs_materialized": 0,
        "quality_predictions_materialized": 0,
        "quality_outcomes_materialized": 0,
        "quality_aggregates_materialized": 0,
        "claim_scope": "trusted-canonical-v1.2-pipeline-only",
        "owner_deletion_or_filesystem_rollback_in_scope": False,
        "out_of_band_or_alternate_output_root_access_in_scope": False,
        "seed_values_are_secret_or_unpublished": False,
        "evaluation_seed_used_to_initialize_quality_rng": False,
        "quality_rng_initialized": False,
    }
    result = _attested_payload(
        payload,
        trust_root=trust_root,
        purpose=NONOBSERVATION_PURPOSE,
    )
    _assert_canonical_nonobservation_paths_absent(repository_root=quality_context.repository_root)
    _assert_prospective_quality_paths_absent(repository_root=quality_context.repository_root)
    return result


def _verify_nonobservation(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    quality_context: QualityContext,
    historical_receipt: Mapping[str, Any],
    recheck_paths: bool,
) -> None:
    expected_fields = {
        "schema_version",
        "artifact_type",
        "experiment_id",
        "status",
        "admission_nonce",
        "quality_source",
        "quality_manifest",
        "historical_receipt_payload_sha256",
        "historical_result_source_commit",
        "evaluation_seed_namespace",
        "evaluation_seed_namespace_reselected",
        "canonical_absent_paths",
        "prospective_quality_absent_paths",
        "canonical_quality_output_count",
        "canonical_worker_ledger_count",
        "canonical_quality_matrix_lock_count",
        "canonical_integrity_output_count",
        "canonical_summary_output_count",
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "claim_scope",
        "owner_deletion_or_filesystem_rollback_in_scope",
        "out_of_band_or_alternate_output_root_access_in_scope",
        "seed_values_are_secret_or_unpublished",
        "evaluation_seed_used_to_initialize_quality_rng",
        "quality_rng_initialized",
        "payload_sha256",
        "attestation",
    }
    _require(set(payload) == expected_fields, "Canonical non-observation schema drifted.")
    _verify_attested_payload(
        payload,
        trust_root=trust_root,
        purpose=NONOBSERVATION_PURPOSE,
        label="Canonical non-observation",
    )
    expected_paths = tuple(
        str(_absolute(path, repository_root=quality_context.repository_root))
        for path in CANONICAL_NONOBSERVATION_PATHS
    )
    expected_prospective_paths = tuple(
        str(_absolute(path, repository_root=quality_context.repository_root))
        for path in PROSPECTIVE_QUALITY_PATHS
    )
    _require(
        payload.get("schema_version") == NONOBSERVATION_SCHEMA_VERSION
        and payload.get("artifact_type") == "direct-controller-canonical-nonobservation"
        and payload.get("experiment_id") == quality_context.manifest_binding["experiment_id"]
        and payload.get("status") == "terminal"
        and _is_sha256(payload.get("admission_nonce"))
        and payload.get("quality_source") == quality_context.source
        and payload.get("quality_manifest") == quality_context.manifest_binding
        and payload.get("historical_receipt_payload_sha256")
        == historical_receipt.get("payload_sha256")
        and payload.get("historical_result_source_commit") == HISTORICAL_RESULT_SOURCE_COMMIT
        and tuple(payload.get("evaluation_seed_namespace", ())) == EVALUATION_SEEDS
        and payload.get("evaluation_seed_namespace_reselected") is False
        and tuple(payload.get("canonical_absent_paths", ())) == expected_paths
        and tuple(payload.get("prospective_quality_absent_paths", ())) == expected_prospective_paths
        and all(
            payload.get(field) == 0
            for field in (
                "canonical_quality_output_count",
                "canonical_worker_ledger_count",
                "canonical_quality_matrix_lock_count",
                "canonical_integrity_output_count",
                "canonical_summary_output_count",
                "evaluation_inputs_materialized",
                "quality_predictions_materialized",
                "quality_outcomes_materialized",
                "quality_aggregates_materialized",
            )
        )
        and payload.get("claim_scope") == "trusted-canonical-v1.2-pipeline-only"
        and payload.get("owner_deletion_or_filesystem_rollback_in_scope") is False
        and payload.get("out_of_band_or_alternate_output_root_access_in_scope") is False
        and payload.get("seed_values_are_secret_or_unpublished") is False
        and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and payload.get("quality_rng_initialized") is False,
        "Canonical non-observation contract drifted.",
    )
    if recheck_paths:
        _assert_canonical_nonobservation_paths_absent(
            repository_root=quality_context.repository_root
        )
        _assert_prospective_quality_paths_absent(repository_root=quality_context.repository_root)


def _sealed_bytes_fd(name: str, payload: bytes) -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(create is not None and allow_sealing is not None, "Sealed memfd support is required.")
    descriptor = cast(Any, create)(
        name,
        int(getattr(os, "MFD_CLOEXEC", 0)) | int(cast(int, allow_sealing)),
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "Sealed snapshot write stalled.")
            offset += written
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        _require(
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
            "Snapshot memfd was not sealed.",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _writable_receipt_fd() -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(create is not None and allow_sealing is not None, "Receipt memfd support is required.")
    return cast(Any, create)(
        "adaptive-v4-v1-3-historical-receipt",
        int(getattr(os, "MFD_CLOEXEC", 0)) | int(cast(int, allow_sealing)),
    )


@contextmanager
def _detached_historical_worktree(repository_root: Path) -> Iterator[Path]:
    parent = Path(tempfile.mkdtemp(prefix="adaptive-v4-v1-3-worktree-"))
    os.chmod(parent, SAFE_DIRECTORY_MODE)
    worktree = parent / "result-source-8c884"
    added = False
    try:
        result = _git(
            repository_root,
            ["worktree", "add", "--detach", str(worktree), HISTORICAL_RESULT_SOURCE_COMMIT],
            check=False,
        )
        _require(result.returncode == 0, "Detached historical worktree could not be created.")
        added = True
        state = _source_state(worktree)
        _require(
            state == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Detached historical worktree is not the exact clean result source.",
        )
        _assert_tree_object(
            worktree,
            HISTORICAL_RESULT_SOURCE_COMMIT,
            HISTORICAL_RESULT_SOURCE_TREE,
        )
        yield worktree
        _require(
            _source_state(worktree) == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Detached historical worktree changed during validation.",
        )
    finally:
        if added:
            removed = _git(
                repository_root,
                ["worktree", "remove", "--force", str(worktree)],
                check=False,
            )
            pruned = _git(repository_root, ["worktree", "prune"], check=False)
            _require(
                removed.returncode == 0
                and pruned.returncode == 0
                and not os.path.lexists(worktree),
                "Detached historical worktree cleanup failed closed.",
            )
        try:
            parent.rmdir()
        except OSError as error:
            raise ValueError(
                "Detached historical worktree parent was not empty after cleanup."
            ) from error


def _historical_manifest_binding(repository_root: Path) -> dict[str, Any]:
    path = _exact_path(
        HISTORICAL_MANIFEST_RELATIVE_PATH,
        label="Historical v1.2 manifest",
        repository_root=repository_root,
        must_exist=True,
    )
    payload, opened = _load_json_nofollow(path, label="Historical v1.2 manifest")
    try:
        _require(
            opened.sha256 == HISTORICAL_MANIFEST_SHA256
            and opened.bytes == HISTORICAL_MANIFEST_BYTES
            and payload.get("experiment_id") == HISTORICAL_MANIFEST_EXPERIMENT_ID,
            "Historical v1.2 manifest binding drifted.",
        )
        committed = _git(
            repository_root,
            [
                "show",
                f"{HISTORICAL_RESULT_SOURCE_COMMIT}:{HISTORICAL_MANIFEST_RELATIVE_PATH.as_posix()}",
            ],
            text=False,
        ).stdout
        _require(
            committed == opened.read_bytes(),
            "Live historical manifest path differs from the result-source blob.",
        )
        opened.assert_unchanged()
        return {
            "path": str(path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
            "experiment_id": HISTORICAL_MANIFEST_EXPERIMENT_ID,
        }
    finally:
        opened.close()


def _expected_historical_ledger_bindings(repository_root: Path) -> dict[str, dict[str, Any]]:
    specs = {
        "training": (HISTORICAL_TRAINING_LEDGER, HISTORICAL_TRAINING_LEDGER_SHA256),
        "calibration": (HISTORICAL_CALIBRATION_LEDGER, HISTORICAL_CALIBRATION_LEDGER_SHA256),
        "top_p": (HISTORICAL_TOP_P_LEDGER, HISTORICAL_TOP_P_LEDGER_SHA256),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (relative_path, expected_sha256) in specs.items():
        path = _exact_path(
            relative_path,
            label=f"Historical {name} ledger",
            repository_root=repository_root,
            must_exist=True,
        )
        payload, opened = _load_json_nofollow(path, label=f"Historical {name} ledger")
        try:
            _require(opened.sha256 == expected_sha256, f"Historical {name} ledger hash drifted.")
            result[name] = {
                "path": str(path),
                "sha256": opened.sha256,
                "bytes": opened.bytes,
                "experiment_id": payload.get("experiment_id"),
                "payload_sha256": payload.get("payload_sha256"),
                "attestation_mac": (
                    payload.get("attestation", {}).get("mac")
                    if isinstance(payload.get("attestation"), Mapping)
                    else None
                ),
            }
            opened.assert_unchanged()
        finally:
            opened.close()
    return result


def _archived_validation_payload(
    *,
    trust_root: attestation.TrustRoot,
    repository_root: Path,
) -> dict[str, Any]:
    # Imported here so importing this admission module cannot materialize quality coordinates,
    # touch CUDA, or import either the old or new controller evaluator.
    import run_p2_direct_top_p_physical_matrix as top_p_matrix

    detached_state = _source_state(Path.cwd())
    _require(
        detached_state == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
        "Archived child is not running in the exact detached result source.",
    )
    _assert_tree_object(Path.cwd(), HISTORICAL_RESULT_SOURCE_COMMIT, HISTORICAL_RESULT_SOURCE_TREE)
    _assert_tree_object(
        Path.cwd(),
        HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
        HISTORICAL_IMPLEMENTATION_SOURCE_TREE,
    )
    _assert_tree_object(Path.cwd(), V1_1_RESULT_SOURCE_COMMIT, V1_1_RESULT_SOURCE_TREE)
    _assert_tree_object(
        Path.cwd(), V1_1_IMPLEMENTATION_SOURCE_COMMIT, V1_1_IMPLEMENTATION_SOURCE_TREE
    )
    _assert_ancestor(
        Path.cwd(), HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT, HISTORICAL_RESULT_SOURCE_COMMIT
    )
    _assert_ancestor(Path.cwd(), V1_1_IMPLEMENTATION_SOURCE_COMMIT, V1_1_RESULT_SOURCE_COMMIT)
    _assert_ancestor(Path.cwd(), V1_1_RESULT_SOURCE_COMMIT, HISTORICAL_RESULT_SOURCE_COMMIT)
    inventory = _assert_live_inventory_matches_historical(repository_root)
    manifest_binding = _historical_manifest_binding(repository_root)

    # The legacy public loader expects a path-based trust-root loader. The key is intentionally
    # absent from the child environment; replace only that transport function with the already
    # validated sealed-FD trust root. No scientific predicate or artifact validator is changed.
    def sealed_transport(
        *,
        repository_root: Path,
        artifact_roots: tuple[Path, ...] = (),
        expected_key_id: str | None = None,
    ) -> attestation.TrustRoot:
        del repository_root, artifact_roots
        _require(
            expected_key_id is None or expected_key_id == trust_root.key_id,
            "Legacy loader requested a different attestation trust root.",
        )
        return trust_root

    attestation.trust_root_from_environment = cast(Any, sealed_transport)

    prerequisites = top_p_matrix.load_and_validate_prerequisites(
        manifest_path=repository_root / HISTORICAL_MANIFEST_RELATIVE_PATH,
        training_output_root=repository_root / HISTORICAL_TRAINING_ROOT,
        calibration_output_root=repository_root / HISTORICAL_CALIBRATION_ROOT,
        output_root=repository_root / HISTORICAL_TOP_P_ROOT,
        attestation_key_path=None,
    )
    _require(
        prerequisites.context.source == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
        "Archived prerequisite context is not bound to 8c884.",
    )
    generator = top_p_matrix._canonical_generator(top_p_matrix.GENERATOR_SCRIPT)
    opened_generator, generator_snapshot = top_p_matrix._open_generator(generator)
    opened_generator.close()
    top_p_path = repository_root / HISTORICAL_TOP_P_LEDGER
    top_p_payload, top_p_opened = _load_json_nofollow(
        top_p_path, label="Historical top-p matrix ledger"
    )
    try:
        records = top_p_matrix.validate_matrix_summary_archived(
            top_p_payload,
            output_root=repository_root / HISTORICAL_TOP_P_ROOT,
            prerequisites=prerequisites,
            generator_script=generator,
            generator_binding=generator_snapshot.public_binding,
            matrix_lock_binding=top_p_matrix._matrix_lock_binding(
                top_p_matrix._matrix_lock_path(repository_root / HISTORICAL_TOP_P_ROOT)
            ),
            verify_artifacts=True,
        )
        top_p_matrix._preflight_output_tree(
            output_root=repository_root / HISTORICAL_TOP_P_ROOT,
            matrix_summary=top_p_path,
            completed_cells=len(records),
        )
        _require(
            top_p_opened.sha256 == HISTORICAL_TOP_P_LEDGER_SHA256
            and top_p_payload.get("status") == "terminal"
            and top_p_payload.get("terminal_decision") == "NO-GO"
            and top_p_payload.get("completed_cells") == 40
            and top_p_payload.get("go_cells") == 0
            and top_p_payload.get("no_go_cells") == 40
            and len(records) == 40
            and all(record.get("terminal_decision") == "NO-GO" for record in records),
            "Historical top-p matrix is not the exact terminal 0/40 GO result.",
        )
        top_p_opened.assert_unchanged()
    finally:
        top_p_opened.close()

    calibrations: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for scale in SCALES:
        for training_seed in TRAINING_SEEDS:
            item = prerequisites.calibrations[(scale, training_seed)]
            payload = dict(item.payload)
            observation = payload.get("quality_evaluation_started")
            _require(
                payload.get("source") == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False}
                and payload.get("terminal_decision") == "GO"
                and all(
                    payload.get("budget_decisions", {}).get(budget) == "GO" for budget in BUDGETS
                )
                and observation in {None, False},
                f"Historical admitted calibration is not all-GO: {scale}/{training_seed}.",
            )
            checkpoint = payload.get("checkpoint")
            _require(isinstance(checkpoint, Mapping), "Calibration checkpoint binding is missing.")
            artifact_binding = dict(item.binding)
            artifact_binding["attestation_purpose"] = LEGACY_CALIBRATION_PURPOSE
            calibrations.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "calibration_seed": CALIBRATION_SEEDS[TRAINING_SEEDS.index(training_seed)],
                    "artifact": artifact_binding,
                    "checkpoint": dict(cast(Mapping[str, Any], checkpoint)),
                }
            )
            checkpoints.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "checkpoint": dict(cast(Mapping[str, Any], checkpoint)),
                }
            )
    _require(
        len(calibrations) == len(checkpoints) == 10, "Historical input inventory is incomplete."
    )

    ledgers = _expected_historical_ledger_bindings(repository_root)
    training_payload, training_opened = _load_json_nofollow(
        repository_root / HISTORICAL_TRAINING_LEDGER,
        label="Historical training ledger",
    )
    try:
        _require(
            training_payload.get("status") == "terminal"
            and training_payload.get("completed_runs") == 10
            and training_payload.get("expected_runs") == 10,
            "Historical training matrix is not terminal 10/10.",
        )
        training_environment = training_payload.get("execution_environment")
        _require(
            isinstance(training_environment, Mapping),
            "Historical training execution environment is missing.",
        )
        training_opened.assert_unchanged()
    finally:
        training_opened.close()

    calibration_payload, calibration_opened = _load_json_nofollow(
        repository_root / HISTORICAL_CALIBRATION_LEDGER,
        label="Historical calibration ledger",
    )
    try:
        _require(
            calibration_payload.get("status") == "terminal"
            and calibration_payload.get("terminal_decision") == "GO"
            and calibration_payload.get("completed_cells") == 10
            and calibration_payload.get("expected_cells") == 10
            and calibration_payload.get("quality_evaluation_started") is False,
            "Historical calibration matrix is not terminal 10/10 GO pre-quality evidence.",
        )
        calibration_opened.assert_unchanged()
    finally:
        calibration_opened.close()

    closed_world = {
        "training": _scan_closed_world_root(
            repository_root / HISTORICAL_TRAINING_ROOT,
            label="Historical training root",
            expected_files=HISTORICAL_ROOT_COUNTS["training"]["files"],
            expected_directories=HISTORICAL_ROOT_COUNTS["training"]["directories"],
        ),
        "calibration_quarantine": _scan_closed_world_root(
            repository_root / HISTORICAL_CALIBRATION_QUARANTINE_ROOT,
            label="Historical calibration quarantine",
            expected_files=HISTORICAL_ROOT_COUNTS["calibration-quarantine"]["files"],
            expected_directories=HISTORICAL_ROOT_COUNTS["calibration-quarantine"]["directories"],
        ),
        "calibration": _scan_closed_world_root(
            repository_root / HISTORICAL_CALIBRATION_ROOT,
            label="Historical calibration root",
            expected_files=HISTORICAL_ROOT_COUNTS["calibration-v1-2"]["files"],
            expected_directories=HISTORICAL_ROOT_COUNTS["calibration-v1-2"]["directories"],
        ),
        "top_p": _scan_closed_world_root(
            repository_root / HISTORICAL_TOP_P_ROOT,
            label="Historical top-p root",
            expected_files=HISTORICAL_ROOT_COUNTS["top-p"]["files"],
            expected_directories=HISTORICAL_ROOT_COUNTS["top-p"]["directories"],
        ),
    }
    absent = _assert_canonical_nonobservation_paths_absent(repository_root=repository_root)
    module_origin_audit = _audit_loaded_repo_module_origins(
        repository_root=repository_root,
        detached_root=Path.cwd(),
        allowed_inventory=_commit_inventory(
            repository_root,
            HISTORICAL_RESULT_SOURCE_COMMIT,
            HISTORICAL_IMPLEMENTATION_PATHS,
        ),
        sealed_admission=_sealed_admission_module_binding(),
    )
    return _attested_payload(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "artifact_type": "direct-controller-historical-validation-receipt",
            "status": "terminal",
            "historical_result_source": {
                "commit": HISTORICAL_RESULT_SOURCE_COMMIT,
                "tree": HISTORICAL_RESULT_SOURCE_TREE,
                "dirty": False,
            },
            "historical_implementation": {
                "source_commit": HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
                "tree": HISTORICAL_IMPLEMENTATION_SOURCE_TREE,
                "digest": HISTORICAL_IMPLEMENTATION_DIGEST,
            },
            "historical_manifest": manifest_binding,
            "historical_runtime_inventory": inventory,
            "attestation_key_id": trust_root.key_id,
            "ledgers": ledgers,
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
            "training_execution_environment": dict(cast(Mapping[str, Any], training_environment)),
            "closed_world": closed_world,
            "canonical_v1_2_absent_paths": list(absent),
            "evaluation_seed_namespace": list(EVALUATION_SEEDS),
            "evaluation_seed_namespace_reselected": False,
            "evaluation_seed_used_to_initialize_quality_rng": False,
            "quality_rng_initialized": False,
            "validation_mode": "stored-only-detached-result-source-no-cuda",
            "legacy_key_transport_override": "sealed-fd-only-loader-adapter",
            "module_origin_audit": module_origin_audit,
            "scientific_subprocesses_started": 0,
            "quality_evaluator_imported": False,
            "evaluation_generator_imported": False,
            "gpu_or_cuda_api_accessed": False,
        },
        trust_root=trust_root,
        purpose=HISTORICAL_RECEIPT_PURPOSE,
    )


def _write_fd_payload(file_descriptor: int, payload: Mapping[str, Any]) -> None:
    encoded = canonical_json(payload)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    os.ftruncate(file_descriptor, 0)
    offset = 0
    while offset < len(encoded):
        written = os.write(file_descriptor, encoded[offset:])
        _require(written > 0, "Archived receipt write stalled.")
        offset += written
    os.fsync(file_descriptor)


def _archived_child_entry(arguments: argparse.Namespace) -> int:
    repository_root = _exact_path(
        Path(arguments.repository_root), label="Canonical repository root", must_exist=True
    )
    _require(Path.cwd() != repository_root, "Archived child requires a detached worktree cwd.")
    raw_receipt_fd = os.environ.pop(ARCHIVED_RECEIPT_FD_ENV, None)
    _require(raw_receipt_fd is not None and raw_receipt_fd.isdigit(), "Receipt FD is missing.")
    trust_root = attestation.trust_root_from_inherited_environment(
        expected_key_id=arguments.expected_key_id
    )
    payload = _archived_validation_payload(
        trust_root=trust_root,
        repository_root=repository_root,
    )
    _write_fd_payload(int(cast(str, raw_receipt_fd)), payload)
    return 0


def create_historical_validation_receipt(
    *,
    trust_root: attestation.TrustRoot,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    root = _exact_path(repository_root, label="Repository root", must_exist=True)
    _assert_tree_object(root, HISTORICAL_RESULT_SOURCE_COMMIT, HISTORICAL_RESULT_SOURCE_TREE)
    _assert_tree_object(
        root,
        HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
        HISTORICAL_IMPLEMENTATION_SOURCE_TREE,
    )
    _assert_ancestor(root, HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT, HISTORICAL_RESULT_SOURCE_COMMIT)
    inventory_before = _assert_live_inventory_matches_historical(root)
    manifest_before = _historical_manifest_binding(root)
    ledger_before = _expected_historical_ledger_bindings(root)
    _assert_canonical_nonobservation_paths_absent(repository_root=root)

    module_path = Path(__file__).resolve(strict=True)
    opened_module = _open_secure_regular(module_path, label="v1.3 reuse-admission module")
    runtime_fd = -1
    module_fd = -1
    import_inventory_fd = -1
    key_fd = -1
    receipt_fd = -1
    try:
        runtime_binding = _archived_python_runtime_binding(root)
        runtime_fd = _sealed_bytes_fd(
            "adaptive-v4-v1-3-archived-python-runtime",
            canonical_json(runtime_binding),
        )
        module_bytes = opened_module.read_bytes()
        module_fd = _sealed_bytes_fd("adaptive-v4-v1-3-admission-module", module_bytes)
        import_inventory_fd = _sealed_bytes_fd(
            "adaptive-v4-v1-3-archived-import-inventory",
            canonical_json(_historical_import_inventory_payload(root)),
        )
        key_fd = attestation.create_sealed_key_fd(trust_root)
        receipt_fd = _writable_receipt_fd()
        environment = _archived_child_environment(
            runtime_fd=runtime_fd,
            module_fd=module_fd,
            import_inventory_fd=import_inventory_fd,
            key_fd=key_fd,
            receipt_fd=receipt_fd,
        )
        with _detached_historical_worktree(root) as worktree:
            pycache_prefix = worktree.parent / "isolated-empty-pycache"
            pycache_prefix.mkdir(mode=SAFE_DIRECTORY_MODE)
            os.chmod(pycache_prefix, SAFE_DIRECTORY_MODE)
            _require(not any(pycache_prefix.iterdir()), "Isolated pycache prefix is not empty.")
            try:
                command = _archived_child_command(
                    runtime_binding=runtime_binding,
                    pycache_prefix=pycache_prefix,
                    module_path=module_path,
                    repository_root=root,
                    expected_key_id=trust_root.key_id,
                )
                completed = subprocess.run(
                    command,
                    cwd=worktree,
                    env=environment,
                    pass_fds=(runtime_fd, module_fd, import_inventory_fd, key_fd, receipt_fd),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                _require(
                    completed.returncode == 0,
                    "Archived historical validation child failed without an admissible receipt.",
                )
            finally:
                _require(
                    not any(pycache_prefix.iterdir()),
                    "Archived child populated its isolated no-bytecode cache prefix.",
                )
                pycache_prefix.rmdir()
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(receipt_fd, fcntl.F_ADD_SEALS, seals)
        raw = attestation.read_all_fd(receipt_fd, maximum_bytes=16 * 1024 * 1024)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Archived child receipt is not valid JSON.") from error
        _require(isinstance(payload, dict), "Archived child receipt must be a JSON object.")
        _require(raw == canonical_json(payload), "Archived child receipt bytes are not canonical.")
        _verify_historical_receipt(cast(Mapping[str, Any], payload), trust_root=trust_root)
        opened_module.assert_unchanged()
    finally:
        if receipt_fd >= 0:
            os.close(receipt_fd)
        if key_fd >= 0:
            os.close(key_fd)
        if import_inventory_fd >= 0:
            os.close(import_inventory_fd)
        if module_fd >= 0:
            os.close(module_fd)
        if runtime_fd >= 0:
            os.close(runtime_fd)
        opened_module.close()

    _require(
        _assert_live_inventory_matches_historical(root) == inventory_before,
        "Historical runtime inventory changed around the archived child.",
    )
    _require(_historical_manifest_binding(root) == manifest_before, "Historical manifest changed.")
    _require(
        _expected_historical_ledger_bindings(root) == ledger_before,
        "Historical ledger bindings changed around the archived child.",
    )
    _assert_canonical_nonobservation_paths_absent(repository_root=root)
    return cast(dict[str, Any], payload)


def _declared_historical_inventory_path(path: str) -> bool:
    return any(
        path == declared or path.startswith(f"{declared.rstrip('/')}/")
        for declared in HISTORICAL_IMPLEMENTATION_PATHS
    )


def _verify_module_origin_audit(payload: object) -> None:
    _require(isinstance(payload, Mapping), "Historical module-origin audit is missing.")
    audit = cast(Mapping[str, Any], payload)
    _require(
        set(audit)
        == {
            "semantics",
            "count",
            "digest",
            "origins",
            "sealed_admission",
            "third_party_runtime_boundary",
        }
        and audit.get("semantics") == MODULE_ORIGIN_AUDIT_SEMANTICS
        and _is_sha256(audit.get("digest"))
        and isinstance(audit.get("origins"), list)
        and isinstance(audit.get("sealed_admission"), Mapping),
        "Historical module-origin audit schema drifted.",
    )
    rows = cast(list[Any], audit["origins"])
    sealed = cast(Mapping[str, Any], audit["sealed_admission"])
    third_party_runtime_boundary = audit.get("third_party_runtime_boundary")
    _require(bool(rows), "Historical module-origin audit is empty.")
    seen: set[tuple[str, str]] = set()
    expected_row_fields = {
        "module",
        "origin",
        "relative_path",
        "source_location",
        "git_mode",
        "git_blob_oid",
        "sha256",
        "bytes",
    }
    for raw in rows:
        _require(isinstance(raw, Mapping), "Historical module-origin row is invalid.")
        row = cast(Mapping[str, Any], raw)
        module = row.get("module")
        origin = row.get("origin")
        relative_path = row.get("relative_path")
        source_location = row.get("source_location")
        _require(
            set(row) == expected_row_fields
            and isinstance(module, str)
            and bool(module)
            and isinstance(origin, str)
            and isinstance(relative_path, str)
            and relative_path.endswith(".py")
            and _declared_historical_inventory_path(relative_path)
            and source_location in {"canonical-live", "detached-result-source"}
            and origin == f"{source_location}:{relative_path}"
            and row.get("git_mode") in {"100644", "100755"}
            and _is_git_oid(row.get("git_blob_oid"))
            and _is_sha256(row.get("sha256"))
            and type(row.get("bytes")) is int
            and cast(int, row["bytes"]) > 0,
            "Historical module-origin row drifted.",
        )
        key = (cast(str, module), cast(str, origin))
        _require(key not in seen, "Historical module-origin row is duplicated.")
        seen.add(key)
    _require(
        set(sealed) == {"path", "sha256", "bytes", "transport"}
        and sealed.get("path") == str(Path(__file__).resolve(strict=True))
        and _is_sha256(sealed.get("sha256"))
        and type(sealed.get("bytes")) is int
        and cast(int, sealed["bytes"]) > 0
        and sealed.get("transport") == "sealed-memfd-exec"
        and isinstance(third_party_runtime_boundary, Mapping)
        and dict(cast(Mapping[str, Any], third_party_runtime_boundary))
        == _archived_third_party_runtime_boundary(_archived_python_runtime_binding(REPOSITORY_ROOT))
        and audit.get("count") == len(rows) + 1
        and audit.get("digest")
        == _json_digest(
            {
                "origins": rows,
                "sealed_admission": dict(sealed),
                "third_party_runtime_boundary": dict(
                    cast(Mapping[str, Any], third_party_runtime_boundary)
                ),
            }
        ),
        "Historical sealed admission or module-origin digest drifted.",
    )
    opened = attestation.open_regular_nofollow(Path(cast(str, sealed["path"])))
    try:
        metadata = os.fstat(opened.file_descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and not bool(stat.S_IMODE(metadata.st_mode) & stat.S_IWOTH)
            and opened.sha256 == sealed["sha256"]
            and opened.bytes == sealed["bytes"],
            "Live admission module differs from the sealed archived-child snapshot.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()


def _verify_historical_receipt(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
) -> None:
    expected_fields = {
        "schema_version",
        "artifact_type",
        "status",
        "historical_result_source",
        "historical_implementation",
        "historical_manifest",
        "historical_runtime_inventory",
        "attestation_key_id",
        "ledgers",
        "training",
        "calibration",
        "top_p",
        "calibrations",
        "checkpoints",
        "training_execution_environment",
        "closed_world",
        "canonical_v1_2_absent_paths",
        "evaluation_seed_namespace",
        "evaluation_seed_namespace_reselected",
        "evaluation_seed_used_to_initialize_quality_rng",
        "quality_rng_initialized",
        "validation_mode",
        "legacy_key_transport_override",
        "module_origin_audit",
        "scientific_subprocesses_started",
        "quality_evaluator_imported",
        "evaluation_generator_imported",
        "gpu_or_cuda_api_accessed",
        "payload_sha256",
        "attestation",
    }
    _require(set(payload) == expected_fields, "Historical validation receipt schema drifted.")
    _verify_attested_payload(
        payload,
        trust_root=trust_root,
        purpose=HISTORICAL_RECEIPT_PURPOSE,
        label="Historical validation receipt",
    )
    result_source = payload.get("historical_result_source")
    implementation = payload.get("historical_implementation")
    ledgers = payload.get("ledgers")
    training = payload.get("training")
    calibration = payload.get("calibration")
    top_p = payload.get("top_p")
    calibrations = payload.get("calibrations")
    checkpoints = payload.get("checkpoints")
    _verify_module_origin_audit(payload.get("module_origin_audit"))
    _require(
        payload.get("schema_version") == RECEIPT_SCHEMA_VERSION
        and payload.get("artifact_type") == "direct-controller-historical-validation-receipt"
        and payload.get("status") == "terminal"
        and result_source
        == {
            "commit": HISTORICAL_RESULT_SOURCE_COMMIT,
            "tree": HISTORICAL_RESULT_SOURCE_TREE,
            "dirty": False,
        }
        and implementation
        == {
            "source_commit": HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
            "tree": HISTORICAL_IMPLEMENTATION_SOURCE_TREE,
            "digest": HISTORICAL_IMPLEMENTATION_DIGEST,
        }
        and payload.get("historical_manifest", {}).get("sha256") == HISTORICAL_MANIFEST_SHA256
        and payload.get("attestation_key_id") == trust_root.key_id
        and isinstance(ledgers, Mapping)
        and cast(Mapping[str, Any], ledgers).get("training", {}).get("sha256")
        == HISTORICAL_TRAINING_LEDGER_SHA256
        and cast(Mapping[str, Any], ledgers).get("calibration", {}).get("sha256")
        == HISTORICAL_CALIBRATION_LEDGER_SHA256
        and cast(Mapping[str, Any], ledgers).get("top_p", {}).get("sha256")
        == HISTORICAL_TOP_P_LEDGER_SHA256
        and training == {"completed_runs": 10, "expected_runs": 10, "status": "terminal"}
        and calibration
        == {
            "completed_cells": 10,
            "expected_cells": 10,
            "terminal_decision": "GO",
            "quality_evaluation_started": False,
        }
        and isinstance(top_p, Mapping)
        and cast(Mapping[str, Any], top_p).get("completed_cells") == 40
        and cast(Mapping[str, Any], top_p).get("go_cells") == 0
        and cast(Mapping[str, Any], top_p).get("no_go_cells") == 40
        and cast(Mapping[str, Any], top_p).get("terminal_decision") == "NO-GO"
        and cast(Mapping[str, Any], top_p).get("calibration_only_cells") == 40
        and cast(Mapping[str, Any], top_p).get("evaluation_seed_accessed_cells") == 0
        and cast(Mapping[str, Any], top_p).get("reuse_role") == "historical-disclosure-only"
        and cast(Mapping[str, Any], top_p).get("quality_gate_eligible") is False
        and cast(Mapping[str, Any], top_p).get("quality_input_eligible") is False
        and cast(Mapping[str, Any], top_p).get("statistical_input_eligible") is False
        and isinstance(calibrations, list)
        and len(calibrations) == 10
        and isinstance(checkpoints, list)
        and len(checkpoints) == 10
        and tuple(payload.get("evaluation_seed_namespace", ())) == EVALUATION_SEEDS
        and payload.get("evaluation_seed_namespace_reselected") is False
        and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and payload.get("quality_rng_initialized") is False
        and payload.get("validation_mode") == "stored-only-detached-result-source-no-cuda"
        and payload.get("legacy_key_transport_override") == "sealed-fd-only-loader-adapter"
        and payload.get("scientific_subprocesses_started") == 0
        and payload.get("quality_evaluator_imported") is False
        and payload.get("evaluation_generator_imported") is False
        and payload.get("gpu_or_cuda_api_accessed") is False,
        "Historical validation receipt contract drifted.",
    )
    coordinates: set[tuple[str, int]] = set()
    for raw in cast(list[Any], calibrations):
        _require(isinstance(raw, Mapping), "Receipt calibration entry is invalid.")
        item = cast(Mapping[str, Any], raw)
        coordinate = (item.get("scale"), item.get("training_seed"))
        _require(
            coordinate[0] in SCALES
            and coordinate[1] in TRAINING_SEEDS
            and coordinate not in coordinates
            and item.get("calibration_seed")
            == CALIBRATION_SEEDS[TRAINING_SEEDS.index(cast(int, coordinate[1]))]
            and isinstance(item.get("artifact"), Mapping)
            and isinstance(item.get("checkpoint"), Mapping),
            "Receipt calibration coordinate or binding is invalid.",
        )
        coordinates.add(cast(tuple[str, int], coordinate))
    _require(
        coordinates == {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS},
        "Receipt calibration grid is incomplete.",
    )


def _execution_environment_projection(raw: Mapping[str, Any]) -> dict[str, Any]:
    module = importlib.import_module("adaptive_v4_execution_environment")
    raw_projector = getattr(module, "controller_compatible_environment_projection", None)
    _require(callable(raw_projector), "Execution-environment projection helper is unavailable.")
    projector = cast(Callable[[Mapping[str, Any]], object], raw_projector)
    projected = projector(raw)
    _require(isinstance(projected, Mapping), "Execution-environment projection is invalid.")
    return dict(cast(Mapping[str, Any], projected))


def build_reuse_admission_payload(
    *,
    historical_receipt: Mapping[str, Any],
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    admission_path: Path = DEFAULT_ADMISSION_PATH,
    admission_nonce: str | None = None,
) -> dict[str, Any]:
    _require(type(quality_context) is QualityContext, "Quality context must be canonical.")
    assert_quality_context_unchanged(quality_context)
    _verify_historical_receipt(historical_receipt, trust_root=trust_root)
    _require(
        quality_context.manifest_binding["attestation"]["key_id"] == trust_root.key_id,
        "Quality manifest and historical evidence use different trust roots.",
    )
    nonce = secrets.token_hex(32) if admission_nonce is None else admission_nonce
    _require(_is_sha256(nonce), "Admission nonce must contain 256 random bits.")
    path = _exact_path(
        admission_path,
        label="Canonical v1.3 reuse-admission path",
        repository_root=quality_context.repository_root,
        must_exist=False,
    )
    canonical_default = _absolute(
        DEFAULT_ADMISSION_PATH, repository_root=quality_context.repository_root
    )
    _require(path == canonical_default, "Reuse admission must use its canonical v1.3 path.")
    _require(
        not os.path.lexists(path),
        "Reuse-admission creation refuses an existing final path.",
    )
    nonobservation = build_canonical_nonobservation(
        trust_root=trust_root,
        quality_context=quality_context,
        historical_receipt=historical_receipt,
        admission_nonce=nonce,
    )
    training_environment = historical_receipt.get("training_execution_environment")
    _require(
        isinstance(training_environment, Mapping),
        "Historical receipt training environment is missing.",
    )
    projection = _execution_environment_projection(cast(Mapping[str, Any], training_environment))
    payload = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "artifact_type": "direct-controller-historical-reuse-admission",
        "experiment_id": quality_context.manifest_binding["experiment_id"],
        "status": "terminal",
        "canonical_path": str(path),
        "admission_nonce": nonce,
        "quality_source": quality_context.source,
        "quality_manifest": quality_context.manifest_binding,
        "historical_validation_receipt": dict(historical_receipt),
        "canonical_nonobservation": nonobservation,
        "historical_training_ledger": dict(
            cast(Mapping[str, Any], historical_receipt["ledgers"])["training"]
        ),
        "historical_calibration_ledger": dict(
            cast(Mapping[str, Any], historical_receipt["ledgers"])["calibration"]
        ),
        "historical_top_p_ledger": dict(
            cast(Mapping[str, Any], historical_receipt["ledgers"])["top_p"]
        ),
        "calibrations": list(cast(list[Any], historical_receipt["calibrations"])),
        "checkpoints": list(cast(list[Any], historical_receipt["checkpoints"])),
        "execution_environment_projection": projection,
        "training_seeds": list(TRAINING_SEEDS),
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "evaluation_seed_namespace_reselected": False,
        "quality_evaluation_started": False,
        "evaluation_seed_used_to_initialize_quality_rng": False,
        "quality_rng_initialized": False,
        "evaluation_inputs_materialized": 0,
        "quality_predictions_materialized": 0,
        "quality_outcomes_materialized": 0,
        "quality_aggregates_materialized": 0,
        "top_p_reuse_role": "historical-disclosure-only",
        "top_p_quality_input_count": 0,
        "top_p_quality_gate_eligible": False,
        "top_p_statistical_input_eligible": False,
        "upstream_context_is_quality_context": False,
        "historical_artifacts_rewritten_or_reattested": False,
        "scientific_subprocesses_started_during_admission": 0,
    }
    return _attested_payload(
        payload,
        trust_root=trust_root,
        purpose=REUSE_ADMISSION_PURPOSE,
    )


def _admission_entries(
    payload: Mapping[str, Any],
    *,
    quality_context: QualityContext,
) -> tuple[
    dict[tuple[str, int], AdmittedCalibration],
    dict[tuple[str, int], AdmittedCheckpoint],
]:
    raw_calibrations = payload.get("calibrations")
    raw_checkpoints = payload.get("checkpoints")
    _require(isinstance(raw_calibrations, list), "Admission calibration inventory is missing.")
    _require(isinstance(raw_checkpoints, list), "Admission checkpoint inventory is missing.")
    calibrations: dict[tuple[str, int], AdmittedCalibration] = {}
    checkpoints: dict[tuple[str, int], AdmittedCheckpoint] = {}
    for raw in cast(list[Any], raw_calibrations):
        _require(isinstance(raw, Mapping), "Admission calibration entry is invalid.")
        item = cast(Mapping[str, Any], raw)
        scale = item.get("scale")
        training_seed = item.get("training_seed")
        artifact = item.get("artifact")
        checkpoint = item.get("checkpoint")
        coordinate = (scale, training_seed)
        _require(
            scale in SCALES
            and training_seed in TRAINING_SEEDS
            and coordinate not in calibrations
            and item.get("calibration_seed")
            == CALIBRATION_SEEDS[TRAINING_SEEDS.index(cast(int, training_seed))]
            and isinstance(artifact, Mapping)
            and isinstance(checkpoint, Mapping),
            "Admission calibration coordinate or binding drifted.",
        )
        artifact_map = dict(cast(Mapping[str, Any], artifact))
        checkpoint_map = dict(cast(Mapping[str, Any], checkpoint))
        _require(
            artifact_map.get("attestation_purpose") == LEGACY_CALIBRATION_PURPOSE,
            "Admission calibration attestation purpose drifted.",
        )
        calibration_path = _exact_path(
            Path(cast(str, artifact_map.get("path", ""))),
            label="Admitted calibration path",
            repository_root=quality_context.repository_root,
            must_exist=False,
        )
        checkpoint_path = _exact_path(
            Path(cast(str, checkpoint_map.get("path", ""))),
            label="Admitted checkpoint path",
            repository_root=quality_context.repository_root,
            must_exist=False,
        )
        key = cast(tuple[str, int], coordinate)
        calibrations[key] = AdmittedCalibration(
            path=calibration_path,
            public_binding=artifact_map,
            checkpoint_binding=checkpoint_map,
        )
        checkpoints[key] = AdmittedCheckpoint(
            path=checkpoint_path,
            public_binding=checkpoint_map,
        )
    for raw in cast(list[Any], raw_checkpoints):
        _require(isinstance(raw, Mapping), "Admission checkpoint entry is invalid.")
        item = cast(Mapping[str, Any], raw)
        scale = item.get("scale")
        training_seed = item.get("training_seed")
        valid_key = scale in SCALES and training_seed in TRAINING_SEEDS
        key = (cast(str, scale), cast(int, training_seed))
        _require(
            valid_key
            and key in checkpoints
            and item.get("checkpoint") == checkpoints[key].public_binding,
            "Admission checkpoint duplicate view drifted.",
        )
    expected = {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS}
    _require(
        set(calibrations) == set(checkpoints) == expected,
        "Admission historical input grid is incomplete.",
    )
    return calibrations, checkpoints


def _verify_historical_evidence_against_receipt(
    payload: Mapping[str, Any],
    *,
    quality_context: QualityContext,
) -> None:
    receipt = cast(Mapping[str, Any], payload["historical_validation_receipt"])
    _require(
        _assert_live_inventory_matches_historical(quality_context.repository_root)
        == receipt.get("historical_runtime_inventory"),
        "Live historical runtime inventory differs from the signed receipt.",
    )
    _require(
        _historical_manifest_binding(quality_context.repository_root)
        == receipt.get("historical_manifest"),
        "Historical manifest differs from the signed receipt.",
    )
    _require(
        _expected_historical_ledger_bindings(quality_context.repository_root)
        == receipt.get("ledgers"),
        "Historical ledgers differ from the signed receipt.",
    )
    closed_world = receipt.get("closed_world")
    _require(isinstance(closed_world, Mapping), "Signed historical closed-world view is missing.")
    current_closed_world = {
        "training": _scan_closed_world_root(
            quality_context.repository_root / HISTORICAL_TRAINING_ROOT,
            label="Historical training root",
            expected_files=24,
            expected_directories=13,
        ),
        "calibration_quarantine": _scan_closed_world_root(
            quality_context.repository_root / HISTORICAL_CALIBRATION_QUARANTINE_ROOT,
            label="Historical calibration quarantine",
            expected_files=3,
            expected_directories=3,
        ),
        "calibration": _scan_closed_world_root(
            quality_context.repository_root / HISTORICAL_CALIBRATION_ROOT,
            label="Historical calibration root",
            expected_files=12,
            expected_directories=13,
        ),
        "top_p": _scan_closed_world_root(
            quality_context.repository_root / HISTORICAL_TOP_P_ROOT,
            label="Historical top-p root",
            expected_files=41,
            expected_directories=33,
        ),
    }
    _require(
        current_closed_world == dict(cast(Mapping[str, Any], closed_world)),
        "Historical closed-world roots drifted.",
    )


def _reuse_admission_public_binding(
    *,
    path: Path,
    payload: Mapping[str, Any],
    sha256: str,
    byte_count: int,
) -> dict[str, Any]:
    receipt = payload.get("historical_validation_receipt")
    nonobservation = payload.get("canonical_nonobservation")
    envelope = payload.get("attestation")
    _require(
        isinstance(receipt, Mapping)
        and isinstance(nonobservation, Mapping)
        and isinstance(envelope, Mapping),
        "Reuse-admission public binding inputs are invalid.",
    )
    return {
        "path": str(path),
        "sha256": sha256,
        "bytes": byte_count,
        "experiment_id": payload.get("experiment_id"),
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": cast(Mapping[str, Any], envelope).get("mac"),
        "historical_receipt_sha256": cast(Mapping[str, Any], receipt).get("payload_sha256"),
        "canonical_nonobservation_sha256": cast(Mapping[str, Any], nonobservation).get(
            "payload_sha256"
        ),
    }


def _validate_reuse_admission_internal(
    payload: Mapping[str, Any],
    *,
    admission_path: Path,
    storage_path: Path,
    require_final_root: bool,
    trust_root: attestation.TrustRoot,
    quality_context: QualityContext,
    verify_evidence: bool = True,
) -> ValidatedReuseAdmission:
    _require(type(quality_context) is QualityContext, "Quality context must be canonical.")
    assert_quality_context_unchanged(quality_context)
    canonical_admission_path = _absolute(
        DEFAULT_ADMISSION_PATH,
        repository_root=quality_context.repository_root,
    )
    if require_final_root:
        canonical_admission_path, _canonical_genesis_path = _validate_final_admission_root(
            repository_root=quality_context.repository_root
        )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "experiment_id",
        "status",
        "canonical_path",
        "admission_nonce",
        "quality_source",
        "quality_manifest",
        "historical_validation_receipt",
        "canonical_nonobservation",
        "historical_training_ledger",
        "historical_calibration_ledger",
        "historical_top_p_ledger",
        "calibrations",
        "checkpoints",
        "execution_environment_projection",
        "training_seeds",
        "calibration_seeds",
        "evaluation_seeds",
        "evaluation_seed_namespace_reselected",
        "quality_evaluation_started",
        "evaluation_seed_used_to_initialize_quality_rng",
        "quality_rng_initialized",
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "top_p_reuse_role",
        "top_p_quality_input_count",
        "top_p_quality_gate_eligible",
        "top_p_statistical_input_eligible",
        "upstream_context_is_quality_context",
        "historical_artifacts_rewritten_or_reattested",
        "scientific_subprocesses_started_during_admission",
        "payload_sha256",
        "attestation",
    }
    _require(set(payload) == expected_fields, "Reuse-admission schema drifted.")
    _verify_attested_payload(
        payload,
        trust_root=trust_root,
        purpose=REUSE_ADMISSION_PURPOSE,
        label="Reuse admission",
    )
    receipt = payload.get("historical_validation_receipt")
    nonobservation = payload.get("canonical_nonobservation")
    _require(isinstance(receipt, Mapping), "Historical receipt is missing from admission.")
    _require(isinstance(nonobservation, Mapping), "Canonical non-observation is missing.")
    receipt_map = cast(Mapping[str, Any], receipt)
    nonobservation_map = cast(Mapping[str, Any], nonobservation)
    _verify_historical_receipt(receipt_map, trust_root=trust_root)
    _verify_nonobservation(
        nonobservation_map,
        trust_root=trust_root,
        quality_context=quality_context,
        historical_receipt=receipt_map,
        recheck_paths=True,
    )
    path = _exact_path(
        admission_path,
        label="Canonical reuse-admission path",
        repository_root=quality_context.repository_root,
        must_exist=require_final_root,
    )
    _require(
        path == canonical_admission_path and payload.get("canonical_path") == str(path),
        "Reuse admission path is not canonical.",
    )
    _require(
        payload.get("schema_version") == ADMISSION_SCHEMA_VERSION
        and payload.get("artifact_type") == "direct-controller-historical-reuse-admission"
        and payload.get("experiment_id") == quality_context.manifest_binding["experiment_id"]
        and payload.get("status") == "terminal"
        and _is_sha256(payload.get("admission_nonce"))
        and payload.get("quality_source") == quality_context.source
        and payload.get("quality_manifest") == quality_context.manifest_binding
        and payload.get("historical_training_ledger")
        == cast(Mapping[str, Any], receipt_map["ledgers"])["training"]
        and payload.get("historical_calibration_ledger")
        == cast(Mapping[str, Any], receipt_map["ledgers"])["calibration"]
        and payload.get("historical_top_p_ledger")
        == cast(Mapping[str, Any], receipt_map["ledgers"])["top_p"]
        and tuple(payload.get("training_seeds", ())) == TRAINING_SEEDS
        and tuple(payload.get("calibration_seeds", ())) == CALIBRATION_SEEDS
        and tuple(payload.get("evaluation_seeds", ())) == EVALUATION_SEEDS
        and payload.get("evaluation_seed_namespace_reselected") is False
        and payload.get("quality_evaluation_started") is False
        and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and payload.get("quality_rng_initialized") is False
        and all(
            payload.get(name) == 0
            for name in (
                "evaluation_inputs_materialized",
                "quality_predictions_materialized",
                "quality_outcomes_materialized",
                "quality_aggregates_materialized",
                "top_p_quality_input_count",
                "scientific_subprocesses_started_during_admission",
            )
        )
        and payload.get("top_p_reuse_role") == "historical-disclosure-only"
        and payload.get("top_p_quality_gate_eligible") is False
        and payload.get("top_p_statistical_input_eligible") is False
        and payload.get("upstream_context_is_quality_context") is False
        and payload.get("historical_artifacts_rewritten_or_reattested") is False,
        "Reuse-admission scientific or provenance contract drifted.",
    )
    expected_projection = _execution_environment_projection(
        cast(Mapping[str, Any], receipt_map["training_execution_environment"])
    )
    _require(
        payload.get("execution_environment_projection") == expected_projection,
        "Admission execution-environment projection drifted.",
    )
    calibrations, checkpoints = _admission_entries(payload, quality_context=quality_context)

    storage = _exact_path(
        storage_path,
        label="Reuse-admission storage path",
        repository_root=quality_context.repository_root,
        must_exist=True,
    )
    on_disk_payload, opened = _load_json_nofollow(
        storage,
        label="Reuse admission",
        require_canonical_pretty_bytes=True,
    )
    try:
        _require(
            on_disk_payload == dict(payload), "Supplied admission differs from exact disk bytes."
        )
        public_binding = _reuse_admission_public_binding(
            path=path,
            payload=payload,
            sha256=opened.sha256,
            byte_count=opened.bytes,
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    result = ValidatedReuseAdmission(
        payload=dict(payload),
        public_binding=public_binding,
        calibrations=calibrations,
        checkpoints=checkpoints,
        quality_context=quality_context,
        execution_environment_projection=expected_projection,
    )
    if verify_evidence:
        _verify_historical_evidence_against_receipt(payload, quality_context=quality_context)
        for coordinate in sorted(calibrations):
            calibration_payload, calibration_opened = _load_json_nofollow(
                calibrations[coordinate].path,
                label="Admitted calibration",
            )
            calibration_opened.close()
            validate_admitted_calibration(
                calibration_payload,
                admission=result,
                trust_root=trust_root,
                expected_scale=coordinate[0],
                expected_training_seed=coordinate[1],
            )
            _validate_file_binding(
                checkpoints[coordinate].public_binding,
                label="Admitted checkpoint",
                repository_root=quality_context.repository_root,
            )
    assert_quality_context_unchanged(quality_context)
    if require_final_root:
        _validate_final_admission_root(repository_root=quality_context.repository_root)
    return result


def validate_reuse_admission(
    payload: Mapping[str, Any],
    *,
    admission_path: Path = DEFAULT_ADMISSION_PATH,
    trust_root: attestation.TrustRoot,
    quality_context: QualityContext,
    verify_evidence: bool = True,
) -> ValidatedReuseAdmission:
    return _validate_reuse_admission_internal(
        payload,
        admission_path=admission_path,
        storage_path=admission_path,
        require_final_root=True,
        trust_root=trust_root,
        quality_context=quality_context,
        verify_evidence=verify_evidence,
    )


def load_validated_reuse_admission(
    path: Path = DEFAULT_ADMISSION_PATH,
    *,
    trust_root: attestation.TrustRoot,
    quality_context: QualityContext,
    verify_evidence: bool = True,
) -> ValidatedReuseAdmission:
    canonical_admission_path, _canonical_genesis_path = _validate_final_admission_root(
        repository_root=quality_context.repository_root
    )
    absolute = _exact_path(
        path,
        label="Reuse admission",
        repository_root=quality_context.repository_root,
        must_exist=True,
    )
    _require(absolute == canonical_admission_path, "Reuse admission path is not canonical.")
    payload, opened = _load_json_nofollow(
        absolute,
        label="Reuse admission",
        require_canonical_pretty_bytes=True,
    )
    opened.close()
    return validate_reuse_admission(
        payload,
        admission_path=absolute,
        trust_root=trust_root,
        quality_context=quality_context,
        verify_evidence=verify_evidence,
    )


def _verify_legacy_calibration_attestation(
    payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
) -> None:
    payload_sha256 = payload.get("payload_sha256")
    _require(_is_sha256(payload_sha256), "Calibration payload digest is invalid.")
    source = dict(payload)
    envelope = source.pop("attestation", None)
    source.pop("payload_sha256", None)
    _require(
        payload_sha256 == _json_digest(source),
        "Calibration payload digest does not match its contents.",
    )
    _require(isinstance(envelope, Mapping), "Calibration attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation", None)
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=LEGACY_CALIBRATION_PURPOSE,
    )


def validate_admitted_calibration(
    calibration: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    admission: ValidatedReuseAdmission,
    trust_root: attestation.TrustRoot,
    expected_scale: str,
    expected_training_seed: int,
) -> dict[str, Any]:
    _require(
        type(admission) is ValidatedReuseAdmission,
        "A raw or duck-typed reuse admission is forbidden.",
    )
    _require(isinstance(calibration, Mapping), "Calibration payload must be a mapping.")
    _require(expected_scale in SCALES, "Expected calibration scale is invalid.")
    _require(expected_training_seed in TRAINING_SEEDS, "Expected training seed is invalid.")
    _require(
        admission.payload.get("quality_evaluation_started") is False
        and admission.payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and admission.payload.get("quality_rng_initialized") is False,
        "Reuse admission does not preserve the pre-held-out boundary.",
    )
    assert_quality_context_unchanged(admission.quality_context)
    coordinate = (expected_scale, expected_training_seed)
    _require(coordinate in admission.calibrations, "Calibration coordinate was not admitted.")
    admitted = admission.calibrations[coordinate]
    requested_path = (
        admitted.path
        if artifact_path is None
        else _exact_path(
            artifact_path,
            label="Requested admitted calibration",
            repository_root=admission.quality_context.repository_root,
            must_exist=True,
        )
    )
    _require(
        requested_path == admitted.path, "Requested calibration path is not the admitted path."
    )
    _validate_file_binding(
        admitted.public_binding,
        label="Admitted calibration",
        repository_root=admission.quality_context.repository_root,
    )
    disk_payload, opened = _load_json_nofollow(admitted.path, label="Admitted calibration")
    try:
        _require(
            disk_payload == dict(calibration), "Calibration argument differs from exact disk bytes."
        )
        _require(
            opened.sha256 == admitted.public_binding.get("sha256")
            and opened.bytes == admitted.public_binding.get("bytes"),
            "Calibration bytes differ from the admitted artifact binding.",
        )
        _verify_legacy_calibration_attestation(disk_payload, trust_root=trust_root)
        manifest = disk_payload.get("manifest")
        _require(
            isinstance(manifest, Mapping), "Calibration historical manifest binding is missing."
        )
        manifest_map = cast(Mapping[str, Any], manifest)
        checkpoint = disk_payload.get("checkpoint")
        _require(
            disk_payload.get("schema_version") == 4
            and disk_payload.get("experiment_id") == "p2-post-rank-direct-soft-lag-calibration-v1"
            and disk_payload.get("artifact_type") == "direct-soft-lag-calibration"
            and disk_payload.get("status") == "terminal"
            and disk_payload.get("terminal_decision") == "GO"
            and disk_payload.get("scale") == expected_scale
            and disk_payload.get("training_seed") == expected_training_seed
            and disk_payload.get("calibration_seed")
            == CALIBRATION_SEEDS[TRAINING_SEEDS.index(expected_training_seed)]
            and disk_payload.get("source")
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False}
            and manifest_map.get("sha256") == HISTORICAL_MANIFEST_SHA256
            and manifest_map.get("experiment_id") == HISTORICAL_MANIFEST_EXPERIMENT_ID
            and manifest_map.get("implementation_source_commit")
            == HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT
            and manifest_map.get("implementation_digest") == HISTORICAL_IMPLEMENTATION_DIGEST
            and manifest_map.get("attestation", {}).get("key_id") == trust_root.key_id
            and all(
                disk_payload.get("budget_decisions", {}).get(budget) == "GO" for budget in BUDGETS
            )
            and checkpoint == admitted.checkpoint_binding,
            "Admitted calibration scientific or historical envelope drifted.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    return disk_payload


def load_admitted_checkpoint(
    calibration: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    admission: ValidatedReuseAdmission,
    trust_root: attestation.TrustRoot,
    expected_scale: str,
    expected_training_seed: int,
) -> Mapping[str, Any]:
    validate_admitted_calibration(
        calibration,
        artifact_path=artifact_path,
        admission=admission,
        trust_root=trust_root,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
    )
    coordinate = (expected_scale, expected_training_seed)
    checkpoint = admission.checkpoints[coordinate]
    _validate_file_binding(
        checkpoint.public_binding,
        label="Admitted checkpoint",
        repository_root=admission.quality_context.repository_root,
    )
    opened = _open_secure_regular(checkpoint.path, label="Admitted checkpoint")
    try:
        import torch

        with opened.duplicate_binary_handle() as handle:
            raw = torch.load(handle, map_location="cpu", weights_only=True)
        _require(isinstance(raw, Mapping), "Admitted checkpoint payload is invalid.")
        opened.assert_unchanged()
        return cast(Mapping[str, Any], raw)
    finally:
        opened.close()


def _validated_exact_fill_arm_names(exact_fill_arm_names: Sequence[str]) -> tuple[str, ...]:
    arms = tuple(exact_fill_arm_names)
    _require(
        bool(arms)
        and len(arms) == len(set(arms))
        and all(
            isinstance(arm, str)
            and bool(arm)
            and arm == arm.strip()
            and "\x00" not in arm
            and "\n" not in arm
            and "\r" not in arm
            and not any(marker in arm.lower() for marker in ("top-p", "top_p", "top p"))
            for arm in arms
        ),
        "Genesis arm inventory must contain unique canonical exact-fill arms only.",
    )
    _require(
        arms == FROZEN_EXACT_FILL_ARM_NAMES,
        "Genesis requires the exact ordered 17-arm frozen contract.",
    )
    return arms


def _validate_quality_genesis_registration(
    *,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> tuple[str, ...]:
    arms = _validated_exact_fill_arm_names(exact_fill_arm_names)
    contract = importlib.import_module("p2_direct_controller_contract_v1_3")
    raw_digest_builder = getattr(contract, "quality_coordinate_digest", None)
    _require(callable(raw_digest_builder), "Canonical quality coordinate digest is unavailable.")
    digest_builder = cast(Callable[[], object], raw_digest_builder)
    contract_arms = tuple(getattr(contract, "ALL_ARM_NAMES", ()))
    contract_shards = getattr(contract, "BUDGET_SHARDS_TOTAL", None)
    contract_digest = digest_builder()
    _require(
        expected_shards == EXPECTED_QUALITY_SHARDS == contract_shards
        and coordinate_digest == QUALITY_COORDINATE_DIGEST == contract_digest
        and arms == FROZEN_EXACT_FILL_ARM_NAMES == contract_arms
        and len(arms) == 17,
        "Genesis shard grid, coordinate digest, or ordered arm freeze drifted.",
    )
    return arms


def build_preheldout_genesis_payload(
    *,
    admission: ValidatedReuseAdmission,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    genesis_path: Path = DEFAULT_GENESIS_PATH,
) -> dict[str, Any]:
    _require(type(admission) is ValidatedReuseAdmission, "Genesis requires a canonical admission.")
    _require(type(expected_shards) is int, "Expected shard count is invalid.")
    _require(_is_sha256(coordinate_digest), "Genesis coordinate digest is invalid.")
    arms = _validate_quality_genesis_registration(
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    assert_quality_context_unchanged(admission.quality_context)
    _assert_canonical_nonobservation_paths_absent(
        repository_root=admission.quality_context.repository_root
    )
    _assert_prospective_quality_paths_absent(
        repository_root=admission.quality_context.repository_root
    )
    path = _exact_path(
        genesis_path,
        label="Canonical pre-heldout genesis path",
        repository_root=admission.quality_context.repository_root,
        must_exist=False,
    )
    _require(
        path
        == _absolute(
            DEFAULT_GENESIS_PATH, repository_root=admission.quality_context.repository_root
        ),
        "Pre-heldout genesis must use its canonical path.",
    )
    return _attested_payload(
        {
            "schema_version": GENESIS_SCHEMA_VERSION,
            "artifact_type": "direct-controller-preheldout-genesis",
            "experiment_id": admission.quality_context.manifest_binding["experiment_id"],
            "status": "in_progress",
            "canonical_path": str(path),
            "quality_source": admission.quality_context.source,
            "quality_manifest": admission.quality_context.manifest_binding,
            "reuse_admission": admission.public_binding,
            "expected_shards": expected_shards,
            "coordinate_digest": coordinate_digest,
            "exact_fill_arm_names": list(arms),
            "records": [],
            "completed_shards": 0,
            "quality_evaluation_started": False,
            "evaluation_seed_used_to_initialize_quality_rng": False,
            "quality_rng_initialized": False,
            "evaluation_inputs_materialized": 0,
            "quality_predictions_materialized": 0,
            "quality_outcomes_materialized": 0,
            "quality_aggregates_materialized": 0,
            "active_claim_count": 0,
            "worker_ledger_count": 0,
            "top_p_quality_input_count": 0,
        },
        trust_root=trust_root,
        purpose=PREHELDOUT_GENESIS_PURPOSE,
    )


def _genesis_public_binding(
    *,
    path: Path,
    payload: Mapping[str, Any],
    sha256: str,
    byte_count: int,
) -> dict[str, Any]:
    envelope = payload.get("attestation")
    reuse_admission = payload.get("reuse_admission")
    _require(
        isinstance(envelope, Mapping) and isinstance(reuse_admission, Mapping),
        "Pre-heldout genesis public binding inputs are invalid.",
    )
    return {
        "path": str(path),
        "sha256": sha256,
        "bytes": byte_count,
        "experiment_id": payload.get("experiment_id"),
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": cast(Mapping[str, Any], envelope).get("mac"),
        "reuse_admission_sha256": cast(Mapping[str, Any], reuse_admission).get("sha256"),
        "expected_shards": payload.get("expected_shards"),
        "coordinate_digest": payload.get("coordinate_digest"),
    }


def _validate_preheldout_genesis_internal(
    payload: Mapping[str, Any],
    *,
    genesis_path: Path,
    storage_path: Path,
    require_final_root: bool,
    revalidate_admission: bool,
    admission: ValidatedReuseAdmission,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> ValidatedPreheldoutGenesis:
    _require(
        type(admission) is ValidatedReuseAdmission,
        "A raw or duck-typed reuse admission is forbidden for genesis.",
    )
    _require(type(expected_shards) is int, "Expected shard count is invalid.")
    _require(_is_sha256(coordinate_digest), "Genesis coordinate digest is invalid.")
    arms = _validate_quality_genesis_registration(
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    assert_quality_context_unchanged(admission.quality_context)
    canonical_genesis_path = _absolute(
        DEFAULT_GENESIS_PATH,
        repository_root=admission.quality_context.repository_root,
    )
    if require_final_root:
        _canonical_admission_path, canonical_genesis_path = _validate_final_admission_root(
            repository_root=admission.quality_context.repository_root
        )
    if revalidate_admission:
        revalidated_admission = validate_reuse_admission(
            admission.payload,
            admission_path=Path(cast(str, admission.public_binding.get("path", ""))),
            trust_root=trust_root,
            quality_context=admission.quality_context,
            verify_evidence=False,
        )
        _require(
            revalidated_admission.public_binding == admission.public_binding,
            "Genesis reuse-admission instance differs from canonical disk evidence.",
        )
    expected_fields = {
        "schema_version",
        "artifact_type",
        "experiment_id",
        "status",
        "canonical_path",
        "quality_source",
        "quality_manifest",
        "reuse_admission",
        "expected_shards",
        "coordinate_digest",
        "exact_fill_arm_names",
        "records",
        "completed_shards",
        "quality_evaluation_started",
        "evaluation_seed_used_to_initialize_quality_rng",
        "quality_rng_initialized",
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "active_claim_count",
        "worker_ledger_count",
        "top_p_quality_input_count",
        "payload_sha256",
        "attestation",
    }
    _require(set(payload) == expected_fields, "Pre-heldout genesis schema drifted.")
    _verify_attested_payload(
        payload,
        trust_root=trust_root,
        purpose=PREHELDOUT_GENESIS_PURPOSE,
        label="Pre-heldout genesis",
    )
    path = _exact_path(
        genesis_path,
        label="Canonical pre-heldout genesis path",
        repository_root=admission.quality_context.repository_root,
        must_exist=require_final_root,
    )
    zero_fields = (
        "completed_shards",
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "active_claim_count",
        "worker_ledger_count",
        "top_p_quality_input_count",
    )
    _require(
        path == canonical_genesis_path
        and payload.get("schema_version") == GENESIS_SCHEMA_VERSION
        and payload.get("artifact_type") == "direct-controller-preheldout-genesis"
        and payload.get("experiment_id")
        == admission.quality_context.manifest_binding["experiment_id"]
        and payload.get("status") == "in_progress"
        and payload.get("canonical_path") == str(path)
        and payload.get("quality_source") == admission.quality_context.source
        and payload.get("quality_manifest") == admission.quality_context.manifest_binding
        and payload.get("reuse_admission") == admission.public_binding
        and payload.get("expected_shards") == expected_shards
        and payload.get("coordinate_digest") == coordinate_digest
        and tuple(payload.get("exact_fill_arm_names", ())) == arms
        and payload.get("records") == []
        and payload.get("quality_evaluation_started") is False
        and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and payload.get("quality_rng_initialized") is False
        and all(payload.get(field) == 0 for field in zero_fields),
        "Pre-heldout genesis zero-prefix or scientific contract drifted.",
    )
    storage = _exact_path(
        storage_path,
        label="Pre-heldout genesis storage path",
        repository_root=admission.quality_context.repository_root,
        must_exist=True,
    )
    on_disk, opened = _load_json_nofollow(
        storage,
        label="Pre-heldout genesis",
        require_canonical_pretty_bytes=True,
    )
    try:
        _require(on_disk == dict(payload), "Supplied genesis differs from exact disk bytes.")
        public_binding = _genesis_public_binding(
            path=path,
            payload=payload,
            sha256=opened.sha256,
            byte_count=opened.bytes,
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    _assert_canonical_nonobservation_paths_absent(
        repository_root=admission.quality_context.repository_root
    )
    _assert_prospective_quality_paths_absent(
        repository_root=admission.quality_context.repository_root
    )
    assert_quality_context_unchanged(admission.quality_context)
    if require_final_root:
        _validate_final_admission_root(repository_root=admission.quality_context.repository_root)
    return ValidatedPreheldoutGenesis(payload=dict(payload), public_binding=public_binding)


def validate_preheldout_genesis(
    payload: Mapping[str, Any],
    *,
    genesis_path: Path = DEFAULT_GENESIS_PATH,
    admission: ValidatedReuseAdmission,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> ValidatedPreheldoutGenesis:
    return _validate_preheldout_genesis_internal(
        payload,
        genesis_path=genesis_path,
        storage_path=genesis_path,
        require_final_root=True,
        revalidate_admission=True,
        admission=admission,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )


def load_validated_preheldout_genesis(
    path: Path = DEFAULT_GENESIS_PATH,
    *,
    admission: ValidatedReuseAdmission,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> ValidatedPreheldoutGenesis:
    _require(
        type(admission) is ValidatedReuseAdmission,
        "A raw or duck-typed reuse admission is forbidden for genesis.",
    )
    _canonical_admission_path, canonical_genesis_path = _validate_final_admission_root(
        repository_root=admission.quality_context.repository_root
    )
    absolute = _exact_path(
        path,
        label="Pre-heldout genesis",
        repository_root=admission.quality_context.repository_root,
        must_exist=True,
    )
    _require(absolute == canonical_genesis_path, "Pre-heldout genesis path is not canonical.")
    payload, opened = _load_json_nofollow(
        absolute,
        label="Pre-heldout genesis",
        require_canonical_pretty_bytes=True,
    )
    opened.close()
    return validate_preheldout_genesis(
        payload,
        genesis_path=absolute,
        admission=admission,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )


def _fsync_directory(path: Path, *, exact_mode: int | None = None) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Directory durability requires O_NOFOLLOW support.")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | cast(int, no_follow),
    )
    try:
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH) == 0
            and (exact_mode is None or stat.S_IMODE(metadata.st_mode) == exact_mode),
            f"Unsafe publication directory metadata: {path}",
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_durable(path: Path, payload: bytes) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Atomic publication requires O_NOFOLLOW support.")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
        SAFE_FILE_MODE,
    )
    try:
        os.fchmod(descriptor, SAFE_FILE_MODE)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            _require(written > 0, "Atomic publication write stalled.")
            offset += written
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE,
            "Atomic publication file metadata is unsafe.",
        )
    finally:
        os.close(descriptor)


def _remove_safe_staging_directory(path: Path) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) == SAFE_DIRECTORY_MODE,
        f"Unsafe admission staging directory requires manual quarantine: {path}",
    )
    allowed_names = {DEFAULT_ADMISSION_PATH.name, DEFAULT_GENESIS_PATH.name}
    with os.scandir(path) as iterator:
        entries = list(iterator)
    for entry in entries:
        child = Path(entry.path)
        child_metadata = entry.stat(follow_symlinks=False)
        _require(
            entry.name in allowed_names
            and stat.S_ISREG(child_metadata.st_mode)
            and child_metadata.st_uid == os.getuid()
            and child_metadata.st_nlink == 1
            and stat.S_IMODE(child_metadata.st_mode) == SAFE_FILE_MODE,
            f"Unsafe admission staging content requires manual quarantine: {child}",
        )
    for entry in entries:
        os.unlink(entry.path)
    _fsync_directory(path, exact_mode=SAFE_DIRECTORY_MODE)
    os.rmdir(path)


def _recover_staging_only_fail_closed(parent: Path) -> None:
    recovered = False
    with os.scandir(parent) as iterator:
        candidates = sorted(
            (
                Path(entry.path)
                for entry in iterator
                if entry.name.startswith(ADMISSION_STAGING_PREFIX)
            ),
            key=str,
        )
    for candidate in candidates:
        _remove_safe_staging_directory(candidate)
        recovered = True
    if recovered:
        _fsync_directory(parent)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    raw_renameat2 = getattr(library, "renameat2", None)
    _require(raw_renameat2 is not None, "Atomic no-replace directory rename is unavailable.")
    renameat2 = cast(Any, raw_renameat2)
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _provisional_validated_admission(
    payload: Mapping[str, Any],
    *,
    quality_context: QualityContext,
    encoded: bytes,
) -> ValidatedReuseAdmission:
    path = _absolute(DEFAULT_ADMISSION_PATH, repository_root=quality_context.repository_root)
    calibrations, checkpoints = _admission_entries(payload, quality_context=quality_context)
    receipt = payload.get("historical_validation_receipt")
    _require(isinstance(receipt, Mapping), "Historical receipt is missing from admission.")
    training_environment = cast(Mapping[str, Any], receipt).get("training_execution_environment")
    _require(
        isinstance(training_environment, Mapping),
        "Historical receipt training environment is missing.",
    )
    projection = _execution_environment_projection(cast(Mapping[str, Any], training_environment))
    return ValidatedReuseAdmission(
        payload=dict(payload),
        public_binding=_reuse_admission_public_binding(
            path=path,
            payload=payload,
            sha256=hashlib.sha256(encoded).hexdigest(),
            byte_count=len(encoded),
        ),
        calibrations=calibrations,
        checkpoints=checkpoints,
        quality_context=quality_context,
        execution_environment_projection=projection,
    )


def publish_admission_genesis_bundle(
    *,
    historical_receipt: Mapping[str, Any],
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> tuple[ValidatedReuseAdmission, ValidatedPreheldoutGenesis]:
    _require(type(quality_context) is QualityContext, "Quality context must be canonical.")
    _require(type(expected_shards) is int, "Expected shard count is invalid.")
    _require(_is_sha256(coordinate_digest), "Genesis coordinate digest is invalid.")
    arms = _validate_quality_genesis_registration(
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    root = _absolute(ADMISSION_ROOT, repository_root=quality_context.repository_root)
    expected_root = _absolute(
        QUALITY_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-admission",
        repository_root=quality_context.repository_root,
    )
    _require(root == expected_root, "Admission bundle root is not canonical.")
    parent = root.parent
    _require(parent.exists() and parent.is_dir(), "Admission bundle parent must already exist.")
    parent = _exact_path(parent, label="Admission bundle parent", must_exist=True)
    _assert_canonical_nonobservation_paths_absent(repository_root=quality_context.repository_root)
    _assert_prospective_quality_paths_absent(repository_root=quality_context.repository_root)
    from adaptive_v4_gpu_lock import acquire_gpu_lock

    lease = acquire_gpu_lock(
        "p2-direct-controller-v1.3-admission-publication",
        path=DIRECT_GPU_SCHEDULER_LOCK_PATH,
    )
    staging: Path | None = None
    published = False
    try:
        lease.assert_held()
        _fsync_directory(parent)
        _require(
            not os.path.lexists(root),
            "Existing or partial final admission bundle is immutable and cannot be repaired.",
        )
        _recover_staging_only_fail_closed(parent)
        _require(
            not os.path.lexists(root),
            "Final admission bundle appeared during staging recovery.",
        )
        assert_quality_context_unchanged(quality_context)
        _assert_canonical_nonobservation_paths_absent(
            repository_root=quality_context.repository_root
        )
        _assert_prospective_quality_paths_absent(repository_root=quality_context.repository_root)
        _verify_historical_receipt(historical_receipt, trust_root=trust_root)
        admission_payload = build_reuse_admission_payload(
            historical_receipt=historical_receipt,
            quality_context=quality_context,
            trust_root=trust_root,
        )
        admission_encoded = canonical_pretty_json(admission_payload)
        provisional_admission = _provisional_validated_admission(
            admission_payload,
            quality_context=quality_context,
            encoded=admission_encoded,
        )
        genesis_payload = build_preheldout_genesis_payload(
            admission=provisional_admission,
            trust_root=trust_root,
            expected_shards=expected_shards,
            coordinate_digest=coordinate_digest,
            exact_fill_arm_names=arms,
        )
        genesis_encoded = canonical_pretty_json(genesis_payload)
        staging = parent / f"{ADMISSION_STAGING_PREFIX}{secrets.token_hex(16)}"
        os.mkdir(staging, SAFE_DIRECTORY_MODE)
        os.chmod(staging, SAFE_DIRECTORY_MODE)
        _write_exclusive_durable(staging / DEFAULT_ADMISSION_PATH.name, admission_encoded)
        _write_exclusive_durable(staging / DEFAULT_GENESIS_PATH.name, genesis_encoded)
        _fsync_directory(staging, exact_mode=SAFE_DIRECTORY_MODE)
        _validate_admission_bundle_root(staging, label="Staged admission bundle root")
        staged_admission = _validate_reuse_admission_internal(
            admission_payload,
            admission_path=root / DEFAULT_ADMISSION_PATH.name,
            storage_path=staging / DEFAULT_ADMISSION_PATH.name,
            require_final_root=False,
            trust_root=trust_root,
            quality_context=quality_context,
            verify_evidence=True,
        )
        _require(
            staged_admission.public_binding == provisional_admission.public_binding,
            "Fully validated staged admission differs from its genesis binding.",
        )
        staged_genesis = _validate_preheldout_genesis_internal(
            genesis_payload,
            genesis_path=root / DEFAULT_GENESIS_PATH.name,
            storage_path=staging / DEFAULT_GENESIS_PATH.name,
            require_final_root=False,
            revalidate_admission=False,
            admission=staged_admission,
            trust_root=trust_root,
            expected_shards=expected_shards,
            coordinate_digest=coordinate_digest,
            exact_fill_arm_names=arms,
        )
        lease.assert_held()
        assert_quality_context_unchanged(quality_context)
        _assert_canonical_nonobservation_paths_absent(
            repository_root=quality_context.repository_root
        )
        _assert_prospective_quality_paths_absent(repository_root=quality_context.repository_root)
        _require(
            not os.path.lexists(root),
            "Final admission bundle appeared before no-replace publication.",
        )
        _rename_directory_noreplace(staging, root)
        published = True
        _fsync_directory(parent)
        final_admission_path, final_genesis_path = _validate_final_admission_root(
            repository_root=quality_context.repository_root
        )
        _require(
            final_admission_path == Path(cast(str, staged_admission.public_binding["path"]))
            and final_genesis_path == Path(cast(str, staged_genesis.public_binding["path"])),
            "Published admission bundle paths differ from staged bindings.",
        )
        _validate_file_binding(
            staged_admission.public_binding,
            label="Published reuse admission",
            repository_root=quality_context.repository_root,
        )
        _validate_file_binding(
            staged_genesis.public_binding,
            label="Published pre-heldout genesis",
            repository_root=quality_context.repository_root,
        )
        _assert_canonical_nonobservation_paths_absent(
            repository_root=quality_context.repository_root
        )
        _assert_prospective_quality_paths_absent(repository_root=quality_context.repository_root)
        assert_quality_context_unchanged(quality_context)
        lease.assert_held()
        return staged_admission, staged_genesis
    finally:
        if staging is not None and not published and os.path.lexists(staging):
            _remove_safe_staging_directory(staging)
            _fsync_directory(parent)
        lease.close()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate archived v1.2 evidence for v1.3 reuse.")
    parser.add_argument("--archived-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--repository-root", type=Path, default=REPOSITORY_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument("--expected-key-id", default="", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    if arguments.archived_child:
        _require(
            _is_sha256(arguments.expected_key_id), "Archived child expected key ID is invalid."
        )
        return _archived_child_entry(arguments)
    raise RuntimeError(
        "This module exposes fail-closed admission primitives only; final admission/genesis "
        "publication is forbidden until the final manifest/result-source commit exists."
    )


if __name__ == "__main__":
    raise SystemExit(main())
