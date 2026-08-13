"""Generate and verify a reproducible manual CUDA release-gate receipt.

The verification path intentionally depends only on the Python standard library.
PyTorch is imported lazily, and only while generating a receipt on a CUDA host.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
RECEIPT_KIND = "nano-deepseek-v4.cuda-release-gate"
PASS_STATUS = "passed"
_PYTEST_ISOLATION_ARGUMENTS = (
    "-c",
    "pyproject.toml",
    "--rootdir=.",
    "--noconftest",
)
LOGICAL_COMMAND = (
    "python",
    "-m",
    "pytest",
    "-q",
    *_PYTEST_ISOLATION_ARGUMENTS,
    "-m",
    "gpu",
    "tests/test_accelerator.py",
)
COLLECTION_COMMAND = (
    "python",
    "-m",
    "pytest",
    "--collect-only",
    "-q",
    *_PYTEST_ISOLATION_ARGUMENTS,
    "-m",
    "gpu",
    "tests/test_accelerator.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+(?:[^\s]*)?\Z")
_DRIVER_VERSION_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+\Z")
_DEVICE_UUID_RE = re.compile(r"GPU-[0-9A-Fa-f-]+\Z")


class CudaReleaseGateError(RuntimeError):
    """Raised when CUDA release evidence cannot be generated or verified."""


CommandRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]
EnvironmentProbe = Callable[[], Mapping[str, object]]


def repository_root() -> Path:
    """Return the checkout containing this script."""

    return Path(__file__).resolve().parents[1]


def _canonical_relative_path(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:  # pragma: no cover - defensive invariant
        raise CudaReleaseGateError(f"inventory path escapes repository root: {path}") from error
    value = relative.as_posix()
    parsed = PurePosixPath(value)
    if (
        not value
        or value != parsed.as_posix()
        or parsed.is_absolute()
        or ".." in parsed.parts
        or any(part in {"", "."} for part in parsed.parts)
        or "\\" in value
    ):
        raise CudaReleaseGateError(f"non-canonical inventory path: {value!r}")
    return value


def _inventory_paths(root: Path) -> list[Path]:
    root = root.resolve()
    package = root / "nano_deepseek_v4"
    if not package.is_dir() or package.is_symlink():
        raise CudaReleaseGateError("nano_deepseek_v4 must be a real directory")

    package_sources = [path for path in package.rglob("*.py") if path.is_file()]
    required = [
        package / "py.typed",
        root / "pyproject.toml",
        root / "tests" / "test_accelerator.py",
        root / "scripts" / "cuda_release_gate.py",
    ]
    paths = package_sources + required
    missing = [
        _canonical_relative_path(path, root)
        for path in required
        if not path.is_file() or path.is_symlink()
    ]
    if missing:
        raise CudaReleaseGateError(f"required inventory files are missing: {sorted(missing)}")
    symlinks = [
        _canonical_relative_path(path, root)
        for path in paths
        if path.is_symlink()
    ]
    if symlinks:
        raise CudaReleaseGateError(f"inventory files must not be symlinks: {sorted(symlinks)}")

    by_name = {_canonical_relative_path(path, root): path for path in paths}
    if len(by_name) != len(paths):  # pragma: no cover - impossible without filesystem races
        raise CudaReleaseGateError("inventory contains duplicate paths")
    return [by_name[name] for name in sorted(by_name)]


def build_source_inventory(root: Path | None = None) -> list[dict[str, object]]:
    """Hash the complete source surface covered by the manual CUDA gate."""

    resolved_root = (root or repository_root()).resolve()
    inventory: list[dict[str, object]] = []
    for path in _inventory_paths(resolved_root):
        try:
            content = path.read_bytes()
        except OSError as error:
            raise CudaReleaseGateError(f"could not read inventory file {path.name!r}") from error
        inventory.append(
            {
                "path": _canonical_relative_path(path, resolved_root),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size": len(content),
            }
        )
    return inventory


def source_aggregate_sha256(inventory: Sequence[Mapping[str, object]]) -> str:
    """Return the SHA-256 of the canonical inventory representation."""

    encoded = json.dumps(
        list(inventory),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_command(command: Sequence[str], root: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return subprocess.run(
        list(command),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _command_failure(label: str, result: subprocess.CompletedProcess[str]) -> CudaReleaseGateError:
    combined = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    tail = combined[-2000:] if combined else "no command output"
    return CudaReleaseGateError(f"{label} failed with exit code {result.returncode}:\n{tail}")


def _git_commit(root: Path, runner: CommandRunner) -> str:
    result = runner(("git", "rev-parse", "--verify", "HEAD^{commit}"), root)
    if result.returncode != 0:
        raise _command_failure("git commit lookup", result)
    commit = result.stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise CudaReleaseGateError("git returned a malformed SHA-1 commit")
    return commit


def _git_status(root: Path, runner: CommandRunner) -> str:
    result = runner(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        root,
    )
    if result.returncode != 0:
        raise _command_failure("git worktree inspection", result)
    return result.stdout


def _inventory_commit_mismatches(
    root: Path,
    inventory: Sequence[Mapping[str, object]],
    commit: str,
    runner: CommandRunner,
) -> list[str]:
    """Return covered paths whose exact working bytes are absent from ``commit``."""

    mismatches: list[str] = []
    for entry in inventory:
        path = str(entry["path"])
        committed = runner(("git", "rev-parse", "--verify", f"{commit}:{path}"), root)
        if committed.returncode != 0:
            mismatches.append(path)
            continue
        committed_oid = committed.stdout.strip()
        if _GIT_COMMIT_RE.fullmatch(committed_oid) is None:
            raise CudaReleaseGateError(
                f"git returned a malformed blob ID for inventory path {path!r}"
            )
        working = runner(("git", "hash-object", "--no-filters", "--", path), root)
        if working.returncode != 0:
            raise _command_failure(f"working-tree hash for {path}", working)
        working_oid = working.stdout.strip()
        if _GIT_COMMIT_RE.fullmatch(working_oid) is None:
            raise CudaReleaseGateError(
                f"git returned a malformed working-tree blob ID for {path!r}"
            )
        if working_oid != committed_oid:
            mismatches.append(path)
    return mismatches


def _collect_gpu_node_ids(root: Path, runner: CommandRunner) -> list[str]:
    command = (sys.executable, *COLLECTION_COMMAND[1:], "-p", "no:cacheprovider")
    result = runner(command, root)
    if result.returncode != 0:
        raise _command_failure("CUDA test collection", result)
    prefix = "tests/test_accelerator.py::"
    node_ids = sorted(
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith(prefix)
    )
    if not node_ids:
        raise CudaReleaseGateError("CUDA test collection selected no gpu tests")
    if len(node_ids) != len(set(node_ids)):
        raise CudaReleaseGateError("CUDA test collection returned duplicate node IDs")
    return node_ids


def _parse_junit_counts(path: Path) -> dict[str, int]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise CudaReleaseGateError("CUDA pytest run did not produce valid JUnit XML") from error
    cases = list(root.iter("testcase"))
    skipped = sum(any(child.tag == "skipped" for child in case) for case in cases)
    failed = sum(any(child.tag == "failure" for child in case) for case in cases)
    errors = sum(any(child.tag == "error" for child in case) for case in cases)
    passed = len(cases) - skipped - failed - errors
    if passed < 0:  # pragma: no cover - protects against malformed multi-outcome XML
        raise CudaReleaseGateError("CUDA pytest JUnit outcomes are inconsistent")
    return {
        "selected_count": len(cases),
        "passed_count": passed,
        "skipped_count": skipped,
        "failed_count": failed,
        "error_count": errors,
    }


def _run_gpu_tests(
    root: Path,
    node_ids: Sequence[str],
    runner: CommandRunner,
) -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="nano-deepseek-v4-cuda-") as temporary:
        junit_path = Path(temporary) / "pytest.xml"
        command = (
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *_PYTEST_ISOLATION_ARGUMENTS,
            *node_ids,
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit_path}",
        )
        result = runner(command, root)
        counts = _parse_junit_counts(junit_path)
    if result.returncode != 0:
        raise _command_failure("CUDA pytest run", result)
    expected = len(node_ids)
    if counts != {
        "selected_count": expected,
        "passed_count": expected,
        "skipped_count": 0,
        "failed_count": 0,
        "error_count": 0,
    }:
        raise CudaReleaseGateError(f"CUDA pytest outcomes are not an all-pass run: {counts}")
    return counts


def _normalized_device_uuid(value: object) -> str:
    uuid = str(value).strip()
    if uuid and not uuid.startswith("GPU-"):
        uuid = f"GPU-{uuid}"
    if _DEVICE_UUID_RE.fullmatch(uuid) is None:
        raise CudaReleaseGateError("CUDA device UUID is unavailable or malformed")
    return uuid


def _nvidia_driver_version(device_uuid: str) -> tuple[str, str]:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        raise CudaReleaseGateError("nvidia-smi is required to capture driver provenance") from error
    if result.returncode != 0:
        raise CudaReleaseGateError("nvidia-smi could not capture driver provenance")
    rows = list(csv.reader(io.StringIO(result.stdout)))
    matches = [
        (row[0].strip(), row[1].strip())
        for row in rows
        if len(row) == 2 and row[0].strip() == device_uuid
    ]
    if len(matches) != 1:
        raise CudaReleaseGateError("nvidia-smi did not identify the active CUDA device uniquely")
    uuid, version = matches[0]
    if _DRIVER_VERSION_RE.fullmatch(version) is None:
        raise CudaReleaseGateError("NVIDIA driver version is unavailable or malformed")
    return uuid, version


def probe_cuda_environment() -> dict[str, object]:
    """Collect CUDA provenance, importing PyTorch only on the generation path."""

    try:
        import torch
    except (ImportError, OSError) as error:
        raise CudaReleaseGateError("PyTorch is required to generate a CUDA receipt") from error

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise CudaReleaseGateError("CUDA is required to generate a CUDA release receipt")
    index = int(torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(index)
    device_uuid = _normalized_device_uuid(getattr(properties, "uuid", ""))
    driver_uuid, driver_version = _nvidia_driver_version(device_uuid)
    cuda_version = getattr(torch.version, "cuda", None)
    if not isinstance(cuda_version, str) or not cuda_version.strip():
        raise CudaReleaseGateError("PyTorch does not report its CUDA build version")
    cudnn_version = torch.backends.cudnn.version()

    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "torch": {"version": str(torch.__version__)},
        "cuda": {
            "available": True,
            "build_version": cuda_version,
            "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
        },
        "device": {
            "index": index,
            "name": str(properties.name),
            "uuid": device_uuid,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
        },
        "driver": {"device_uuid": driver_uuid, "version": driver_version},
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise CudaReleaseGateError(f"receipt parent directory does not exist: {parent}")
    serialized = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as error:
        raise CudaReleaseGateError(f"could not write receipt {path.name!r} atomically") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def generate_receipt(
    output: Path,
    *,
    root: Path | None = None,
    allow_dirty: bool = False,
    runner: CommandRunner | None = None,
    environment_probe: EnvironmentProbe | None = None,
) -> dict[str, object]:
    """Run the CUDA gate and atomically write its deterministic receipt."""

    resolved_root = (root or repository_root()).resolve()
    command_runner = runner or _run_command
    probe = environment_probe or probe_cuda_environment
    resolved_output = output.resolve()

    initial_inventory = build_source_inventory(resolved_root)
    inventory_paths = {
        (resolved_root / str(entry["path"])).resolve() for entry in initial_inventory
    }
    if resolved_output in inventory_paths:
        raise CudaReleaseGateError("receipt output must not overwrite a source inventory file")

    initial_status = _git_status(resolved_root, command_runner)
    commit = _git_commit(resolved_root, command_runner)
    commit_mismatches = _inventory_commit_mismatches(
        resolved_root,
        initial_inventory,
        commit,
        command_runner,
    )
    worktree_clean = not initial_status.strip() and not commit_mismatches
    if not worktree_clean and not allow_dirty:
        detail = (
            f"; covered paths not byte-identical at HEAD: {commit_mismatches}"
            if commit_mismatches
            else ""
        )
        raise CudaReleaseGateError(
            "git worktree and covered source bytes must match HEAD; "
            f"--allow-dirty is only for local dogfood{detail}"
        )
    environment = dict(probe())
    node_ids = _collect_gpu_node_ids(resolved_root, command_runner)
    counts = _run_gpu_tests(resolved_root, node_ids, command_runner)

    final_inventory = build_source_inventory(resolved_root)
    if final_inventory != initial_inventory:
        raise CudaReleaseGateError("source inventory changed while the CUDA gate was running")
    if _git_commit(resolved_root, command_runner) != commit:
        raise CudaReleaseGateError("git commit changed while the CUDA gate was running")
    if _git_status(resolved_root, command_runner) != initial_status:
        raise CudaReleaseGateError("git worktree changed while the CUDA gate was running")

    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "status": PASS_STATUS,
        "source": {
            "git_commit": commit,
            "worktree_clean": worktree_clean,
            "inventory": initial_inventory,
            "aggregate_sha256": source_aggregate_sha256(initial_inventory),
        },
        "tests": {
            "collection_command": list(COLLECTION_COMMAND),
            "logical_command": list(LOGICAL_COMMAND),
            "execution_command": [
                "python",
                "-m",
                "pytest",
                "-q",
                *_PYTEST_ISOLATION_ARGUMENTS,
                *node_ids,
            ],
            "selected_node_ids": node_ids,
            **counts,
        },
        "environment": environment,
    }
    _validate_receipt_payload(receipt, resolved_root, permit_dirty=True)
    _atomic_write_json(resolved_output, receipt)
    return receipt


def _require_mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise CudaReleaseGateError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise CudaReleaseGateError(
            f"{label} schema drift: missing={sorted(keys - actual)}, "
            f"unexpected={sorted(actual - keys)}"
        )
    return value


def _require_string(value: object, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise CudaReleaseGateError(f"{label} must be a non-empty canonical string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise CudaReleaseGateError(f"{label} is malformed")
    return value


def _require_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CudaReleaseGateError(f"{label} must be an integer >= {minimum}")
    return value


def _validate_inventory_payload(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list) or not value:
        raise CudaReleaseGateError("source.inventory must be a non-empty JSON array")
    validated: list[dict[str, object]] = []
    for index, raw_entry in enumerate(value):
        entry = _require_mapping(
            raw_entry,
            f"source.inventory[{index}]",
            {"path", "sha256", "size"},
        )
        path = _require_string(entry["path"], f"source.inventory[{index}].path")
        parsed = PurePosixPath(path)
        if (
            path != parsed.as_posix()
            or parsed.is_absolute()
            or ".." in parsed.parts
            or any(part in {"", "."} for part in parsed.parts)
            or "\\" in path
        ):
            raise CudaReleaseGateError(f"source.inventory[{index}].path is non-canonical")
        digest = _require_string(
            entry["sha256"],
            f"source.inventory[{index}].sha256",
            _SHA256_RE,
        )
        size = _require_int(entry["size"], f"source.inventory[{index}].size")
        validated.append({"path": path, "sha256": digest, "size": size})
    paths = [str(entry["path"]) for entry in validated]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise CudaReleaseGateError("source.inventory paths must be unique and sorted")
    return validated


def _validate_environment(value: object) -> None:
    environment = _require_mapping(
        value,
        "environment",
        {"python", "torch", "cuda", "device", "driver"},
    )
    python = _require_mapping(
        environment["python"],
        "environment.python",
        {"implementation", "version"},
    )
    _require_string(python["implementation"], "environment.python.implementation")
    _require_string(python["version"], "environment.python.version", _VERSION_RE)

    torch = _require_mapping(environment["torch"], "environment.torch", {"version"})
    _require_string(torch["version"], "environment.torch.version")

    cuda = _require_mapping(
        environment["cuda"],
        "environment.cuda",
        {"available", "build_version", "cudnn_version"},
    )
    if cuda["available"] is not True:
        raise CudaReleaseGateError("environment.cuda.available must be true")
    _require_string(cuda["build_version"], "environment.cuda.build_version", _VERSION_RE)
    if cuda["cudnn_version"] is not None:
        _require_int(cuda["cudnn_version"], "environment.cuda.cudnn_version", minimum=1)

    device = _require_mapping(
        environment["device"],
        "environment.device",
        {
            "index",
            "name",
            "uuid",
            "compute_capability",
            "total_memory_bytes",
            "multiprocessor_count",
        },
    )
    _require_int(device["index"], "environment.device.index")
    _require_string(device["name"], "environment.device.name")
    device_uuid = _require_string(
        device["uuid"], "environment.device.uuid", _DEVICE_UUID_RE
    )
    capability = device["compute_capability"]
    if not isinstance(capability, list) or len(capability) != 2:
        raise CudaReleaseGateError("environment.device.compute_capability must have two integers")
    _require_int(capability[0], "environment.device.compute_capability[0]")
    _require_int(capability[1], "environment.device.compute_capability[1]")
    _require_int(device["total_memory_bytes"], "environment.device.total_memory_bytes", minimum=1)
    _require_int(
        device["multiprocessor_count"],
        "environment.device.multiprocessor_count",
        minimum=1,
    )

    driver = _require_mapping(
        environment["driver"],
        "environment.driver",
        {"device_uuid", "version"},
    )
    driver_uuid = _require_string(
        driver["device_uuid"], "environment.driver.device_uuid", _DEVICE_UUID_RE
    )
    if driver_uuid != device_uuid:
        raise CudaReleaseGateError("driver and CUDA device UUID provenance do not match")
    _require_string(driver["version"], "environment.driver.version", _DRIVER_VERSION_RE)


def _validate_receipt_payload(
    payload: object,
    root: Path,
    *,
    permit_dirty: bool = False,
) -> dict[str, object]:
    receipt = _require_mapping(
        payload,
        "receipt",
        {"schema_version", "kind", "status", "source", "tests", "environment"},
    )
    if receipt["schema_version"] != SCHEMA_VERSION or type(receipt["schema_version"]) is not int:
        raise CudaReleaseGateError("unsupported CUDA receipt schema_version")
    if receipt["kind"] != RECEIPT_KIND:
        raise CudaReleaseGateError("CUDA receipt kind drift")
    if receipt["status"] != PASS_STATUS:
        raise CudaReleaseGateError("CUDA receipt status is not passed")

    source = _require_mapping(
        receipt["source"],
        "source",
        {"git_commit", "worktree_clean", "inventory", "aggregate_sha256"},
    )
    _require_string(source["git_commit"], "source.git_commit", _GIT_COMMIT_RE)
    if type(source["worktree_clean"]) is not bool:
        raise CudaReleaseGateError("source.worktree_clean must be a boolean")
    if source["worktree_clean"] is not True and not permit_dirty:
        raise CudaReleaseGateError("dirty-generated CUDA receipts are not release evidence")
    inventory = _validate_inventory_payload(source["inventory"])
    aggregate = _require_string(
        source["aggregate_sha256"],
        "source.aggregate_sha256",
        _SHA256_RE,
    )
    if source_aggregate_sha256(inventory) != aggregate:
        raise CudaReleaseGateError("source aggregate does not match the receipt inventory")
    actual_inventory = build_source_inventory(root)
    if actual_inventory != inventory:
        raise CudaReleaseGateError("source inventory drifted from the CUDA receipt")
    if source_aggregate_sha256(actual_inventory) != aggregate:
        raise CudaReleaseGateError("current source aggregate does not match the CUDA receipt")

    tests = _require_mapping(
        receipt["tests"],
        "tests",
        {
            "collection_command",
            "logical_command",
            "execution_command",
            "selected_node_ids",
            "selected_count",
            "passed_count",
            "skipped_count",
            "failed_count",
            "error_count",
        },
    )
    if tests["collection_command"] != list(COLLECTION_COMMAND):
        raise CudaReleaseGateError("CUDA collection command drift")
    if tests["logical_command"] != list(LOGICAL_COMMAND):
        raise CudaReleaseGateError("CUDA logical test command drift")
    node_ids = tests["selected_node_ids"]
    if (
        not isinstance(node_ids, list)
        or not node_ids
        or not all(isinstance(node_id, str) and node_id for node_id in node_ids)
        or node_ids != sorted(node_ids)
        or len(node_ids) != len(set(node_ids))
        or not all(node_id.startswith("tests/test_accelerator.py::") for node_id in node_ids)
    ):
        raise CudaReleaseGateError("selected CUDA node IDs must be non-empty, unique, and sorted")
    expected_execution = [
        "python",
        "-m",
        "pytest",
        "-q",
        *_PYTEST_ISOLATION_ARGUMENTS,
        *node_ids,
    ]
    if tests["execution_command"] != expected_execution:
        raise CudaReleaseGateError("CUDA exact-node execution command drift")
    selected = _require_int(tests["selected_count"], "tests.selected_count", minimum=1)
    passed = _require_int(tests["passed_count"], "tests.passed_count")
    skipped = _require_int(tests["skipped_count"], "tests.skipped_count")
    failed = _require_int(tests["failed_count"], "tests.failed_count")
    errors = _require_int(tests["error_count"], "tests.error_count")
    if selected != len(node_ids) or passed != selected:
        raise CudaReleaseGateError("selected and passed CUDA test counts do not match node IDs")
    if skipped or failed or errors:
        raise CudaReleaseGateError("CUDA release evidence contains skips, failures, or errors")

    _validate_environment(receipt["environment"])
    return dict(receipt)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CudaReleaseGateError(f"receipt contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_json(value: str) -> None:
    raise CudaReleaseGateError(f"receipt contains non-finite JSON value {value}")


def load_receipt(path: Path) -> object:
    """Load strict UTF-8 JSON without accepting duplicate keys or NaN values."""

    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(
            content,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_json,
        )
    except CudaReleaseGateError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CudaReleaseGateError(f"could not read valid CUDA receipt {path.name!r}") from error


def verify_receipt(receipt: Path, *, root: Path | None = None) -> dict[str, object]:
    """Verify a receipt against source bytes without importing PyTorch."""

    resolved_root = (root or repository_root()).resolve()
    return _validate_receipt_payload(load_receipt(receipt), resolved_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="run CUDA tests and write a receipt")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--root", type=Path, default=repository_root())
    generate.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow local dogfood on a dirty tree; generated receipt will not verify",
    )

    verify = subparsers.add_parser("verify", help="verify a receipt without requiring CUDA")
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--root", type=Path, default=repository_root())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "generate":
            generate_receipt(
                args.output,
                root=args.root,
                allow_dirty=args.allow_dirty,
            )
            print(f"wrote CUDA release-gate receipt: {args.output}")
        else:
            verify_receipt(args.receipt, root=args.root)
            print(f"verified CUDA release-gate receipt: {args.receipt}")
    except CudaReleaseGateError as error:
        print(f"CUDA release gate failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
