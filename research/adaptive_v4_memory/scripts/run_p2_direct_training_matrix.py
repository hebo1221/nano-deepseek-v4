from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import adaptive_v4_execution_environment as execution_environment
import adaptive_v4_gpu_lock as gpu_lock
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import torch
import train_m1_associative_recall as trainer

EXPERIMENT_ID = "p2-post-rank-direct-training-v1"
ARTIFACT_TYPE = "training-matrix"
SCHEMA_VERSION = 3
STEPS = 1_000
MINIMUM_STEPS = STEPS
FROZEN_SCALES = ("s55", "s151")
FROZEN_TRAINING_SEEDS = (6071406, 6071407, 6071408, 6071409, 6071410)
OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/training")
MATRIX_SUMMARY = OUTPUT_ROOT / "training-matrix.summary.json"
TRAIN_SCRIPT = Path(__file__).with_name("train_m1_associative_recall.py")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TRAIN_IMPLEMENTATION_PATH = TRAIN_SCRIPT.resolve().relative_to(REPOSITORY_ROOT).as_posix()
GPU_LOCK_IMPLEMENTATION_PATH = (
    Path(gpu_lock.__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
)
TRAINER_FD_ENV = "ADAPTIVE_V4_CANONICAL_TRAINER_FD"
TRAINER_FD_BOOTSTRAP = (
    "import os,sys;"
    "p=sys.argv.pop(1);"
    "sys.argv=[p,*sys.argv[1:]];"
    "sys.path.insert(0,os.path.dirname(p));"
    f"fd=int(os.environ[{TRAINER_FD_ENV!r}]);"
    "data=b'';"
    "\nwhile True:"
    "\n chunk=os.read(fd,1048576)"
    "\n if not chunk: break"
    "\n data+=chunk"
    "\ng={'__name__':'__main__','__file__':p,'__package__':None};"
    "exec(compile(data,p,'exec'),g,g)"
)
SUMMARY_ATTESTATION_PURPOSE = trainer.DIRECT_SUMMARY_ATTESTATION_PURPOSE
MATRIX_ATTESTATION_PURPOSE = "p2-direct-training-matrix-v1"
TRAINING_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "scale",
        "seed",
        "initialization_seed",
        "data_order_seed",
        "training_evaluation_seed",
        "source",
        "command",
        "parameters",
        "auxiliary_parameters",
        "config",
        "task_config",
        "sequence_lengths",
        "training_sequence_lengths",
        "num_queries_per_training_sequence",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "ranking_loss_weight",
        "read_ranking_loss_weight",
        "value_loss_weight",
        "training_topk",
        "steps_completed",
        "stopped_early",
        "history",
        "training_hyperparameters",
        "training_transcript",
        "checkpoint",
        "execution_environment",
        "runtime",
        "direct_training_contract",
        "payload_sha256",
        "attestation",
    }
)
TRAINING_RUNTIME_FIELDS = frozenset(
    {
        "python",
        "torch",
        "device",
        "device_spec",
        "logical_device_index",
        "elapsed_seconds",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    }
)
TRAINING_MATRIX_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "source",
        "manifest",
        "attestation_contract",
        "canonical_trainer",
        "gpu_lease",
        "execution_environment",
        "scales",
        "frozen_training_seeds",
        "steps",
        "minimum_steps",
        "seed_rules",
        "expected_runs",
        "completed_runs",
        "runs",
        "payload_sha256",
        "attestation",
    }
)
FROZEN_HYPERPARAMETERS = trainer.direct_training_hyperparameters()
EVALUATION_STEPS = (1, *range(50, STEPS + 1, 50))
LOCK_NAME = ".p2-direct-training-matrix.lock"
CLAIM_NAME = ".p2-direct-training-cell.claim"
GPU_LEASE_SEMANTICS = "project-persistent-inode-exclusive-whole-matrix-v1"
GPU_LEASE_SCOPE = "before-preflight-through-terminal-validation"
GPU_LEASE_ACQUISITION_ORDER = "gpu-lease-before-matrix-lock-before-cell-claim"
GPU_DEVICE_GUARD_SCOPE = "after-exact-environment-capture-through-terminal-validation"

SEED_RULES = {
    "initialization_seed": "training_seed",
    "data_order_seed": "training_seed+1",
    "training_evaluation_seed": "training_seed+10000",
}


@dataclass(frozen=True)
class FrozenContext:
    manifest_path: Path
    manifest_binding: dict[str, Any]
    source: dict[str, str | bool]


@dataclass(frozen=True)
class CanonicalTrainerSnapshot:
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
            "implementation_path": TRAIN_IMPLEMENTATION_PATH,
        }


@dataclass(frozen=True)
class _MatrixLockLease:
    path: Path
    file_descriptor: int
    device: int
    inode: int

    def assert_held(self) -> None:
        try:
            opened = os.fstat(self.file_descriptor)
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(
                "Training matrix lock path was deleted or became inaccessible while held."
            ) from error
        _require(
            stat.S_ISREG(opened.st_mode),
            "Training matrix lock descriptor is no longer a regular file.",
        )
        _require(
            (opened.st_dev, opened.st_ino) == (self.device, self.inode),
            "Training matrix lock descriptor identity changed while held.",
        )
        _require(
            stat.S_ISREG(current.st_mode)
            and (current.st_dev, current.st_ino) == (self.device, self.inode),
            "Training matrix lock path was replaced while held.",
        )
        _require(
            opened.st_uid == current.st_uid == os.getuid()
            and opened.st_nlink == current.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(current.st_mode) == 0o600,
            "Training matrix lock ownership, link count, or mode is unsafe.",
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_gpu_lock_path(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    _require(not absolute.is_symlink(), "Training GPU lease path may not be a symbolic link.")
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as error:
        raise ValueError("Training GPU lease path cannot be resolved safely.") from error
    _require(
        resolved == absolute,
        "Training GPU lease path must use its exact resolved non-symlink path.",
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
        "Training physical-device guard path does not match the selected GPU.",
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
    _require(isinstance(value, Mapping), "Training matrix GPU lease binding is missing.")
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
        "Training matrix GPU lease binding schema drifted.",
    )
    _require(
        raw.get("path") == str(_canonical_gpu_lock_path(Path(cast(str, raw["path"]))))
        and raw.get("semantics") == GPU_LEASE_SEMANTICS
        and raw.get("scope") == GPU_LEASE_SCOPE
        and raw.get("acquisition_order") == GPU_LEASE_ACQUISITION_ORDER
        and raw.get("child_nested_lease") is False
        and raw.get("portable_identity") == "canonical-path-only-inode-excluded",
        "Training matrix GPU lease semantics drifted.",
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
        "Training matrix selected-device lease binding schema drifted.",
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
        "Training matrix physical-device guard binding drifted.",
    )
    replayed = dict(raw)
    if expected is not None:
        _require(
            dict(raw) == dict(expected),
            "Training matrix GPU lease path or semantics drifted on resume.",
        )
    return replayed


def _canonical_train_script(candidate: Path) -> Path:
    _require(not candidate.is_symlink(), "Training script may not be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
        canonical = TRAIN_SCRIPT.resolve(strict=True)
    except OSError as error:
        raise ValueError("Canonical training script is missing or inaccessible.") from error
    _require(resolved == canonical, "Training script must be the canonical manifest-bound trainer.")
    _require(not canonical.is_symlink(), "Canonical training script may not be a symbolic link.")
    _require(
        TRAIN_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS,
        "Canonical trainer is absent from the implementation inventory.",
    )
    return canonical


def _trainer_snapshot(opened: attestation.OpenedRegularFile) -> CanonicalTrainerSnapshot:
    return CanonicalTrainerSnapshot(
        path=str(opened.path),
        sha256=opened.sha256,
        bytes=opened.bytes,
        device=opened.device,
        inode=opened.inode,
        mode=opened.mode,
        mtime_ns=opened.mtime_ns,
        ctime_ns=opened.ctime_ns,
    )


def _open_canonical_trainer(
    canonical: Path,
    *,
    expected: CanonicalTrainerSnapshot | None = None,
) -> tuple[attestation.OpenedRegularFile, CanonicalTrainerSnapshot]:
    opened = attestation.open_regular_nofollow(canonical)
    snapshot = _trainer_snapshot(opened)
    if expected is not None:
        _require(snapshot == expected, "Canonical trainer changed after frozen preflight.")
    return opened, snapshot


def _sealed_trainer_copy(
    opened: attestation.OpenedRegularFile,
    *,
    expected: CanonicalTrainerSnapshot,
) -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(
        create is not None and allow_sealing is not None,
        "Stable trainer execution requires sealed memfd support.",
    )
    create_memfd = cast(Any, create)
    descriptor = create_memfd(
        "adaptive-v4-canonical-trainer",
        int(getattr(os, "MFD_CLOEXEC", 0)) | int(cast(int, allow_sealing)),
    )
    try:
        payload = opened.read_bytes()
        _require(
            len(payload) == expected.bytes
            and attestation.checksum_bytes(payload) == expected.sha256,
            "Canonical trainer snapshot bytes drifted.",
        )
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        _require(
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
            "Trainer memfd snapshot was not sealed.",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _run_trainer_from_stable_script(
    command: Sequence[str],
    *,
    canonical: Path,
    expected: CanonicalTrainerSnapshot,
    trust_root: attestation.TrustRoot,
    gpu_lease: gpu_lock.GPULockLease,
    device_guard: gpu_lock.GPULockLease,
) -> subprocess.CompletedProcess[Any]:
    gpu_lease.assert_held()
    device_guard.assert_held()
    opened, snapshot = _open_canonical_trainer(canonical, expected=expected)
    trainer_fd = _sealed_trainer_copy(opened, expected=snapshot)
    key_fd = attestation.create_sealed_key_fd(trust_root)
    environment = os.environ.copy()
    environment.pop(attestation.KEY_PATH_ENV, None)
    environment[TRAINER_FD_ENV] = str(trainer_fd)
    environment[attestation.KEY_FD_ENV] = str(key_fd)
    bootstrap_command = [
        command[0],
        "-I",
        "-c",
        TRAINER_FD_BOOTSTRAP,
        str(canonical),
        *command[2:],
    ]
    try:
        result = subprocess.run(
            bootstrap_command,
            check=False,
            pass_fds=(
                trainer_fd,
                key_fd,
                gpu_lease.fileno(),
                device_guard.fileno(),
            ),
            env=environment,
        )
        gpu_lease.assert_held()
        device_guard.assert_held()
        opened.assert_unchanged()
        return result
    finally:
        os.close(key_fd)
        os.close(trainer_fd)
        opened.close()


def _sha256(path: Path) -> str:
    opened = attestation.open_regular_nofollow(path)
    try:
        opened.assert_unchanged()
        return opened.sha256
    finally:
        opened.close()


def _load_json(path: Path) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        payload = json.loads(opened.read_bytes().decode("utf-8"))
        opened.assert_unchanged()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error
    finally:
        opened.close()
    _require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
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
    purpose: str,
) -> dict[str, Any]:
    authenticated = _digest_bound_payload(payload)
    authenticated["attestation"] = attestation.attest_payload(
        authenticated,
        trust_root=trust_root,
        purpose=purpose,
    )
    return authenticated


def _validate_payload_digest(payload: Mapping[str, Any], *, label: str) -> None:
    expected = payload.get("payload_sha256")
    _require(contract.is_sha256(expected), f"{label} payload digest is missing or invalid.")
    digest_source = dict(payload)
    digest_source.pop("attestation", None)
    digest_source.pop("payload_sha256", None)
    _require(
        expected == contract.json_digest(digest_source),
        f"{label} payload digest does not match its contents.",
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
) -> None:
    matrix_lock.assert_held()
    _atomic_write_json(path, payload)
    matrix_lock.assert_held()


def _validated_source_state(state: Mapping[str, Any]) -> dict[str, str | bool]:
    commit = state.get("commit")
    _require(contract.is_git_oid(commit), "Source commit is missing or invalid.")
    _require(state.get("dirty") is False, "Direct training requires clean source.")
    _require(set(state) == {"commit", "dirty"}, "Source-state schema drifted.")
    return {"commit": str(commit), "dirty": False}


def _manifest_binding(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    implementation = manifest.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ValueError("Manifest implementation binding is missing.")
    tree_digest = implementation.get("tree_digest")
    source_commit = implementation.get("source_commit")
    _require(contract.is_sha256(tree_digest), "Manifest implementation digest is invalid.")
    _require(contract.is_git_oid(source_commit), "Manifest implementation commit is invalid.")
    manifest_attestation = manifest.get("attestation")
    _require(
        isinstance(manifest_attestation, Mapping),
        "Manifest attestation binding is missing.",
    )
    manifest_attestation = dict(cast(Mapping[str, Any], manifest_attestation))
    _require(
        manifest_attestation
        == attestation.public_manifest_contract(str(manifest_attestation.get("key_id", ""))),
        "Manifest attestation binding drifted.",
    )
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path) if manifest_sha256 is None else manifest_sha256,
        "experiment_id": manifest.get("experiment_id"),
        "implementation_digest": tree_digest,
        "implementation_source_commit": source_commit,
        "attestation": manifest_attestation,
    }


def establish_frozen_context(manifest_path: Path = contract.MANIFEST_PATH) -> FrozenContext:
    _require(not manifest_path.is_symlink(), "Frozen manifest may not be a symbolic link.")
    manifest_path = manifest_path.resolve()
    opened = attestation.open_regular_nofollow(manifest_path)
    try:
        try:
            raw_manifest = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Frozen direct-controller manifest is invalid JSON.") from error
        _require(
            isinstance(raw_manifest, dict), "Frozen direct-controller manifest must be an object."
        )
        manifest = contract.validate_manifest_payload(raw_manifest, verify_implementation=True)
        opened.assert_unchanged()
        manifest_sha256 = opened.sha256
    finally:
        opened.close()
    source = _validated_source_state(contract.source_state())
    context = FrozenContext(
        manifest_path=manifest_path,
        manifest_binding=_manifest_binding(
            manifest_path,
            manifest,
            manifest_sha256=manifest_sha256,
        ),
        source=source,
    )
    assert_environment_unchanged(context)
    return context


def assert_environment_unchanged(context: FrozenContext) -> None:
    _require(context.manifest_path.is_file(), "Frozen direct-controller manifest disappeared.")
    _require(
        _sha256(context.manifest_path) == context.manifest_binding["sha256"],
        "Frozen direct-controller manifest changed during training.",
    )
    _require(
        contract.implementation_tree_digest() == context.manifest_binding["implementation_digest"],
        "Direct-controller implementation changed during training.",
    )
    _require(
        _validated_source_state(contract.source_state()) == context.source,
        "Source commit changed during training.",
    )


def _seed_values(seed: int) -> dict[str, int]:
    return {
        "initialization_seed": seed,
        "data_order_seed": seed + 1,
        "training_evaluation_seed": seed + 10_000,
    }


def _run_output_dir(output_root: Path, scale: str, seed: int) -> Path:
    return output_root / scale / f"seed-{seed}"


def _training_summary_path(output_root: Path, scale: str, seed: int) -> Path:
    return _run_output_dir(output_root, scale, seed) / f"{scale}-training.summary.json"


def _checkpoint_path(output_root: Path, scale: str, seed: int) -> Path:
    return _run_output_dir(output_root, scale, seed) / f"{scale}-step-{STEPS}.pt"


def build_training_command(
    *,
    train_script: Path,
    output_root: Path,
    scale: str,
    seed: int,
    context: FrozenContext,
    launch_nonce: str,
    trainer_sha256: str,
    device_index: int = 0,
    device_routing_identity: Mapping[str, Any],
) -> list[str]:
    _require(type(device_index) is int and device_index >= 0, "Training device index is invalid.")
    routing_identity = execution_environment.validate_device_routing_identity(
        device_routing_identity
    )
    return [
        sys.executable,
        str(train_script),
        "--experiment-id",
        EXPERIMENT_ID,
        "--scale",
        scale,
        "--seed",
        str(seed),
        "--steps",
        str(STEPS),
        "--minimum-steps",
        str(MINIMUM_STEPS),
        "--batch-size",
        str(FROZEN_HYPERPARAMETERS["batch_size"]),
        "--num-queries",
        str(FROZEN_HYPERPARAMETERS["num_queries"]),
        "--learning-rate",
        str(FROZEN_HYPERPARAMETERS["learning_rate"]),
        "--weight-decay",
        str(FROZEN_HYPERPARAMETERS["weight_decay"]),
        "--ranking-loss-weight",
        str(FROZEN_HYPERPARAMETERS["ranking_loss_weight"]),
        "--read-ranking-loss-weight",
        str(FROZEN_HYPERPARAMETERS["read_ranking_loss_weight"]),
        "--value-loss-weight",
        str(FROZEN_HYPERPARAMETERS["value_loss_weight"]),
        "--gradient-clip-norm",
        str(FROZEN_HYPERPARAMETERS["gradient_clip_norm"]),
        "--training-topk",
        str(FROZEN_HYPERPARAMETERS["training_topk"]),
        "--eval-every",
        str(FROZEN_HYPERPARAMETERS["eval_every"]),
        "--eval-batches",
        str(FROZEN_HYPERPARAMETERS["eval_batches"]),
        "--eval-batch-size",
        str(FROZEN_HYPERPARAMETERS["eval_batch_size"]),
        "--disable-early-stop",
        "--device",
        f"cuda:{device_index}",
        "--expected-device-routing-identity-json",
        json.dumps(routing_identity, sort_keys=True, separators=(",", ":")),
        "--output-dir",
        str(_run_output_dir(output_root, scale, seed)),
        "--source-commit",
        str(context.source["commit"]),
        "--manifest-path",
        str(context.manifest_path),
        "--manifest-experiment-id",
        str(context.manifest_binding["experiment_id"]),
        "--manifest-sha256",
        str(context.manifest_binding["sha256"]),
        "--implementation-digest",
        str(context.manifest_binding["implementation_digest"]),
        "--implementation-source-commit",
        str(context.manifest_binding["implementation_source_commit"]),
        "--attestation-key-id",
        str(context.manifest_binding["attestation"]["key_id"]),
        "--launch-nonce",
        launch_nonce,
        "--trainer-sha256",
        trainer_sha256,
        "--direct-hyperparameters-json",
        trainer.canonical_direct_hyperparameters_json(),
    ]


def _command_option(command: Sequence[Any], option: str) -> str:
    positions = [index for index, item in enumerate(command) if item == option]
    _require(len(positions) == 1, f"Training command must contain exactly one {option}.")
    position = positions[0]
    _require(position + 1 < len(command), f"Training command has no value for {option}.")
    value = command[position + 1]
    _require(isinstance(value, str), f"Training command value for {option} is invalid.")
    return value


def _validate_training_command(
    command: Any,
    *,
    train_script: Path,
    output_dir: Path,
    scale: str,
    seed: int,
    context: FrozenContext,
    launch_nonce: str,
    trainer_sha256: str,
    device_index: int,
    device_routing_identity: Mapping[str, Any],
) -> None:
    _require(isinstance(command, list), "Training command is missing or invalid.")
    _require(len(command) >= 2, "Training command is incomplete.")
    _require(isinstance(command[1], str), "Training script path is invalid.")
    _require(
        Path(command[1]).resolve() == train_script.resolve(),
        "Training summary was produced by a different trainer.",
    )
    expected = {
        "--experiment-id": EXPERIMENT_ID,
        "--scale": scale,
        "--seed": str(seed),
        "--steps": str(STEPS),
        "--minimum-steps": str(MINIMUM_STEPS),
        "--device": f"cuda:{device_index}",
        "--expected-device-routing-identity-json": json.dumps(
            execution_environment.validate_device_routing_identity(device_routing_identity),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for option, value in expected.items():
        _require(
            _command_option(command, option) == value,
            f"Training command binding drifted for {option}.",
        )
    command_output_dir = Path(_command_option(command, "--output-dir"))
    _require(
        command_output_dir.resolve() == output_dir.resolve(),
        "Training command output directory drifted.",
    )
    _require("--no-save" not in command, "Direct training may not disable checkpoints.")
    _require("--disable-early-stop" in command, "Direct training early stopping must be disabled.")
    _require(
        command
        == build_training_command(
            train_script=train_script,
            output_root=output_dir.parent.parent,
            scale=scale,
            seed=seed,
            context=context,
            launch_nonce=launch_nonce,
            trainer_sha256=trainer_sha256,
            device_index=device_index,
            device_routing_identity=device_routing_identity,
        ),
        "Training command contains unregistered arguments or ordering drift.",
    )


def _checkpoint_provenance(
    context: FrozenContext,
    *,
    scale: str,
    seed: int,
    launch_nonce: str,
    trainer_sha256: str,
    transcript_root: str,
    execution_environment_binding: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scale": scale,
        "training_seed": seed,
        **_seed_values(seed),
        "steps_completed": STEPS,
        "minimum_steps": MINIMUM_STEPS,
        "launch_nonce": launch_nonce,
        "canonical_trainer_sha256": trainer_sha256,
        "attestation_key_id": context.manifest_binding["attestation"]["key_id"],
        "training_hyperparameters": FROZEN_HYPERPARAMETERS,
        "transcript_scheme": trainer.DIRECT_TRANSCRIPT_SCHEME,
        "transcript_entry_count": STEPS,
        "transcript_root": transcript_root,
        "source": context.source,
        "manifest_sha256": context.manifest_binding["sha256"],
        "implementation_digest": context.manifest_binding["implementation_digest"],
        "implementation_source_commit": context.manifest_binding["implementation_source_commit"],
        "execution_environment": dict(execution_environment_binding),
    }


def _assert_finite_tensors(value: Any, *, label: str) -> None:
    if isinstance(value, torch.Tensor):
        _require(bool(torch.isfinite(value).all()), f"{label} contains a non-finite tensor.")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tensors(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tensors(child, label=f"{label}[{index}]")


def _validate_optimizer_state(
    optimizer: Any,
    *,
    expected_parameter_count: int,
    expected_state_parameter_ids: set[int],
) -> None:
    _require(isinstance(optimizer, Mapping), "Training optimizer state is missing.")
    _require(set(optimizer) == {"state", "param_groups"}, "Optimizer state schema drifted.")
    state = optimizer.get("state")
    groups = optimizer.get("param_groups")
    _require(isinstance(state, Mapping), "Optimizer per-parameter state is invalid.")
    _require(isinstance(groups, list) and len(groups) == 1, "Optimizer parameter groups drifted.")
    group = groups[0]
    _require(isinstance(group, Mapping), "Optimizer parameter group is invalid.")
    expected_group_fields = {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
        "decoupled_weight_decay",
        "params",
    }
    _require(set(group) == expected_group_fields, "Optimizer parameter-group schema drifted.")
    parameters = group.get("params")
    _require(
        isinstance(parameters, list) and parameters == list(range(expected_parameter_count)),
        "Optimizer parameter inventory drifted.",
    )
    frozen_optimizer = FROZEN_HYPERPARAMETERS["optimizer"]
    _require(
        group.get("lr") == FROZEN_HYPERPARAMETERS["learning_rate"]
        and list(group.get("betas", ())) == frozen_optimizer["betas"]
        and group.get("eps") == frozen_optimizer["eps"]
        and group.get("weight_decay") == FROZEN_HYPERPARAMETERS["weight_decay"]
        and group.get("amsgrad") is False
        and group.get("maximize") is False
        and group.get("foreach") is None
        and group.get("capturable") is False
        and group.get("differentiable") is False
        and group.get("fused") is True
        and group.get("decoupled_weight_decay") is True,
        "Optimizer hyperparameters drifted.",
    )
    _require(set(state) == expected_state_parameter_ids, "Optimizer state inventory drifted.")
    for parameter_id, parameter_state in state.items():
        _require(isinstance(parameter_state, Mapping), "Optimizer parameter state is invalid.")
        step = parameter_state.get("step")
        if isinstance(step, torch.Tensor):
            _require(step.numel() == 1, "Optimizer step tensor is invalid.")
            step_value = float(step.item())
        else:
            _require(isinstance(step, (int, float)), "Optimizer step marker is invalid.")
            step_value = float(step)
        _require(step_value == STEPS, "Optimizer did not complete exactly 1000 steps.")
        _assert_finite_tensors(parameter_state, label=f"optimizer.state.{parameter_id}")


def _validate_checkpoint(
    checkpoint: Any,
    *,
    expected_path: Path,
    context: FrozenContext,
    scale: str,
    seed: int,
    launch_nonce: str,
    trainer_sha256: str,
    transcript_root: str,
    execution_environment_binding: Mapping[str, Any],
    return_raw: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Mapping[str, Any]]:
    _require(isinstance(checkpoint, dict), "Training checkpoint metadata is missing.")
    path_value = checkpoint.get("path")
    _require(isinstance(path_value, str), "Training checkpoint path is invalid.")
    path = Path(path_value)
    _require(not path.is_symlink(), "Training checkpoint may not be a symbolic link.")
    _require(
        Path(os.path.abspath(path)) == Path(os.path.abspath(expected_path)),
        "Training checkpoint path drifted.",
    )
    opened = attestation.open_regular_nofollow(path)
    try:
        byte_count = checkpoint.get("bytes")
        _require(type(byte_count) is int and byte_count > 0, "Checkpoint byte count is invalid.")
        _require(opened.bytes == byte_count, "Checkpoint byte count does not match the file.")
        sha256 = checkpoint.get("sha256")
        _require(contract.is_sha256(sha256), "Checkpoint SHA-256 is invalid.")
        _require(opened.sha256 == sha256, "Checkpoint SHA-256 does not match the file.")
        try:
            with opened.duplicate_binary_handle() as handle:
                raw = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError("Training checkpoint payload cannot be loaded safely.") from error
        opened.assert_unchanged()
    finally:
        opened.close()
    _require(isinstance(raw, Mapping), "Training checkpoint payload is invalid.")
    _require(
        set(raw)
        == {
            "checkpoint_schema_version",
            "provenance",
            "config",
            "model",
            "probe_objective",
            "optimizer",
        },
        "Training checkpoint payload schema drifted.",
    )
    _require(
        raw.get("checkpoint_schema_version") == trainer.DIRECT_CHECKPOINT_SCHEMA_VERSION,
        "Training checkpoint schema version drifted.",
    )
    _require(
        raw.get("config") == asdict(trainer.build_config(scale)),
        "Training checkpoint frozen config drifted.",
    )
    state = raw.get("model")
    _require(
        isinstance(state, Mapping) and bool(state), "Training checkpoint model state is missing."
    )
    _require(
        all(
            isinstance(name, str) and isinstance(tensor, torch.Tensor)
            for name, tensor in state.items()
        ),
        "Training checkpoint model state schema is invalid.",
    )
    try:
        expected_config = trainer.build_config(scale)
        expected_model = trainer.DeepSeekV4ForCausalLM(expected_config)
        expected_model.load_state_dict(dict(state), strict=True, assign=True)
        probe_state = raw.get("probe_objective")
        _require(isinstance(probe_state, Mapping), "Training probe-objective state is missing.")
        expected_probe = trainer.CSAProbeObjective(
            head_dim=expected_config.head_dim,
            query_dim=expected_config.q_lora_rank,
            vocab_size=expected_config.vocab_size,
        )
        expected_probe.load_state_dict(dict(probe_state), strict=True, assign=True)
        model_named_parameters = tuple(expected_model.named_parameters())
        probe_named_parameters = tuple(expected_probe.named_parameters())
        parameter_count = len(model_named_parameters) + len(probe_named_parameters)
        # The frozen direct objective executes the full-sequence path, so cache-only HCA
        # compressors and the unused MTP branch legitimately never acquire AdamW state.
        expected_state_parameter_ids = {
            index
            for index, (name, _parameter) in enumerate(model_named_parameters)
            if not name.startswith("mtp_modules.") and ".self_attn.hca." not in name
        }
        expected_state_parameter_ids.update(range(len(model_named_parameters), parameter_count))
        _assert_finite_tensors(state, label="checkpoint.model")
        _assert_finite_tensors(probe_state, label="checkpoint.probe_objective")
        _validate_optimizer_state(
            raw.get("optimizer"),
            expected_parameter_count=parameter_count,
            expected_state_parameter_ids=expected_state_parameter_ids,
        )
        del expected_model, expected_probe
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Training checkpoint model state does not match the frozen config."
        ) from error
    _require(
        raw.get("provenance")
        == _checkpoint_provenance(
            context,
            scale=scale,
            seed=seed,
            launch_nonce=launch_nonce,
            trainer_sha256=trainer_sha256,
            transcript_root=transcript_root,
            execution_environment_binding=execution_environment_binding,
        ),
        "Training checkpoint internal provenance drifted.",
    )
    metadata = {"path": path_value, "sha256": sha256, "bytes": byte_count}
    if return_raw:
        return metadata, raw
    return metadata


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} must be a finite number.",
    )
    converted = float(value)
    if minimum is not None:
        _require(converted >= minimum, f"{label} fell below its minimum.")
    return converted


def _validate_metric_block(value: Any, *, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} is invalid.")
    expected = {
        *(f"accuracy_length_{length}" for length in trainer.SEQUENCE_LENGTHS),
        "mean_accuracy",
    }
    _require(set(value) == expected, f"{label} schema drifted.")
    for field, raw in value.items():
        metric = _finite_number(raw, label=f"{label}.{field}")
        _require(0.0 <= metric <= 1.0, f"{label}.{field} is outside [0,1].")


def _validate_history(history: Any) -> None:
    _require(isinstance(history, list), "Training history is missing.")
    _require(
        [record.get("step") if isinstance(record, Mapping) else None for record in history]
        == list(EVALUATION_STEPS),
        "Training evaluation cadence drifted.",
    )
    expected_fields = {
        "step",
        "loss",
        "answer_loss",
        "ranking_loss",
        "read_ranking_loss",
        "value_loss",
        "ranking_accuracy",
        "read_ranking_accuracy",
        "value_accuracy",
        "probe_layers",
        "gradient_norm",
        "full_memory",
        "native",
        "local_only",
        "native_minus_local",
        "elapsed_seconds",
    }
    previous_elapsed = -1.0
    for record in history:
        _require(isinstance(record, Mapping), "Training history record is invalid.")
        _require(set(record) == expected_fields, "Training history record schema drifted.")
        for field in (
            "loss",
            "answer_loss",
            "ranking_loss",
            "read_ranking_loss",
            "value_loss",
            "gradient_norm",
        ):
            _finite_number(record.get(field), label=f"history.{field}", minimum=0.0)
        for field in ("ranking_accuracy", "read_ranking_accuracy", "value_accuracy"):
            metric = _finite_number(record.get(field), label=f"history.{field}")
            _require(0.0 <= metric <= 1.0, f"history.{field} is outside [0,1].")
        _require(
            type(record.get("probe_layers")) is int and record["probe_layers"] > 0,
            "Training probe-layer count is invalid.",
        )
        for field in ("full_memory", "native", "local_only"):
            _validate_metric_block(record.get(field), label=f"history.{field}")
        native = float(record["native"]["mean_accuracy"])
        local = float(record["local_only"]["mean_accuracy"])
        _require(
            _finite_number(record.get("native_minus_local"), label="history.native_minus_local")
            == native - local,
            "Training native-minus-local metric drifted.",
        )
        elapsed = _finite_number(
            record.get("elapsed_seconds"), label="history.elapsed_seconds", minimum=0.0
        )
        _require(elapsed >= previous_elapsed, "Training elapsed time moved backwards.")
        previous_elapsed = elapsed


def _validate_transcript(
    transcript: Any,
    *,
    scale: str,
    seed: int,
    launch_nonce: str,
    manifest_sha256: str,
    trainer_sha256: str,
) -> str:
    _require(isinstance(transcript, Mapping), "Training transcript is missing.")
    _require(
        set(transcript) == {"scheme", "entry_count", "entries", "root"},
        "Training transcript schema drifted.",
    )
    _require(
        transcript.get("scheme") == trainer.DIRECT_TRANSCRIPT_SCHEME,
        "Training transcript scheme drifted.",
    )
    entries = transcript.get("entries")
    _require(isinstance(entries, list), "Training transcript entries are invalid.")
    _require(
        transcript.get("entry_count") == len(entries) == STEPS,
        "Training transcript does not contain exactly 1000 steps.",
    )
    root = trainer._initial_transcript_root(
        scale=scale,
        seed=seed,
        launch_nonce=launch_nonce,
        manifest_sha256=manifest_sha256,
        trainer_sha256=trainer_sha256,
    )
    expected_fields = {
        "step",
        "optimizer_step",
        "sequence_length",
        "batch_sha256",
        "loss",
        "answer_loss",
        "ranking_loss",
        "read_ranking_loss",
        "value_loss",
        "gradient_norm",
    }
    for expected_step, entry in enumerate(entries, start=1):
        _require(isinstance(entry, Mapping), "Training transcript entry is invalid.")
        _require(set(entry) == expected_fields, "Training transcript entry schema drifted.")
        _require(
            entry.get("step") == entry.get("optimizer_step") == expected_step,
            "Training transcript step coverage drifted.",
        )
        _require(
            entry.get("sequence_length") in trainer.TRAIN_SEQUENCE_LENGTHS,
            "Training transcript sequence length drifted.",
        )
        _require(
            contract.is_sha256(entry.get("batch_sha256")), "Training batch checksum is invalid."
        )
        for field in (
            "loss",
            "answer_loss",
            "ranking_loss",
            "read_ranking_loss",
            "value_loss",
            "gradient_norm",
        ):
            _finite_number(entry.get(field), label=f"transcript.{field}", minimum=0.0)
        root = trainer._extend_transcript_root(root, entry)
    _require(transcript.get("root") == root, "Training transcript hash chain drifted.")
    return root


def _validate_training_base(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    train_script: Path,
    context: FrozenContext,
    scale: str,
    seed: int,
    launch_nonce: str,
    trainer_sha256: str,
    expected_execution_environment: Mapping[str, Any] | None,
    return_raw_checkpoint: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Mapping[str, Any]]:
    _require(
        set(payload) == TRAINING_SUMMARY_FIELDS,
        "Training summary top-level schema drifted.",
    )
    _require(
        payload.get("schema_version") == trainer.DIRECT_SUMMARY_SCHEMA_VERSION,
        "Training summary schema drifted.",
    )
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong training experiment ID.")
    _require(payload.get("scale") == scale, "Training scale drifted.")
    _require(payload.get("seed") == seed, "Training seed drifted.")
    for name, value in _seed_values(seed).items():
        _require(payload.get(name) == value, f"Training seed rule drifted for {name}.")
    _require(payload.get("source") == context.source, "Training source binding drifted.")
    _require(
        payload.get("config") == asdict(trainer.build_config(scale)),
        "Training config drifted.",
    )
    _require(
        payload.get("training_hyperparameters") == FROZEN_HYPERPARAMETERS,
        "Training hyperparameters drifted.",
    )
    _require(
        payload.get("steps_completed") == STEPS,
        f"Direct training must complete exactly {STEPS} steps.",
    )
    _require(payload.get("stopped_early") is False, "Direct training early stopping is forbidden.")
    raw_environment = payload.get("execution_environment")
    _require(
        isinstance(raw_environment, Mapping),
        "Training summary execution environment is missing.",
    )
    frozen_environment = execution_environment.validate_execution_environment(
        cast(Mapping[str, Any], raw_environment)
    )
    if expected_execution_environment is not None:
        _require(
            frozen_environment
            == execution_environment.validate_execution_environment(expected_execution_environment),
            "Training summary execution environment drifted across the matrix.",
        )
    runtime = payload.get("runtime")
    _require(isinstance(runtime, Mapping), "Training runtime provenance is missing.")
    runtime = cast(Mapping[str, Any], runtime)
    _require(set(runtime) == TRAINING_RUNTIME_FIELDS, "Training runtime schema drifted.")
    selected_device = execution_environment.selected_device_class(frozen_environment)
    _require(
        runtime.get("python") == frozen_environment["python_version"]
        and runtime.get("torch") == frozen_environment["torch_version"]
        and runtime.get("device") == selected_device["name"],
        "Training runtime software or selected-device provenance drifted.",
    )
    _require(
        {
            "device_spec": runtime.get("device_spec"),
            "logical_device_index": runtime.get("logical_device_index"),
        }
        == execution_environment.selected_device_context(frozen_environment),
        "Training runtime logical-device context drifted.",
    )
    _finite_number(runtime.get("elapsed_seconds"), label="runtime.elapsed_seconds", minimum=0.0)
    peak_allocated = runtime.get("peak_allocated_bytes")
    peak_reserved = runtime.get("peak_reserved_bytes")
    _require(
        type(peak_allocated) is int
        and peak_allocated >= 0
        and type(peak_reserved) is int
        and peak_reserved >= peak_allocated,
        "Training runtime CUDA memory counters are invalid.",
    )
    _validate_history(payload.get("history"))
    transcript_root = _validate_transcript(
        payload.get("training_transcript"),
        scale=scale,
        seed=seed,
        launch_nonce=launch_nonce,
        manifest_sha256=str(context.manifest_binding["sha256"]),
        trainer_sha256=trainer_sha256,
    )
    output_dir = _run_output_dir(output_root, scale, seed)
    _validate_training_command(
        payload.get("command"),
        train_script=train_script,
        output_dir=output_dir,
        scale=scale,
        seed=seed,
        context=context,
        launch_nonce=launch_nonce,
        trainer_sha256=trainer_sha256,
        device_index=cast(int, frozen_environment["current_device_index"]),
        device_routing_identity=execution_environment.selected_device_routing_identity(
            frozen_environment
        ),
    )
    return _validate_checkpoint(
        payload.get("checkpoint"),
        expected_path=_checkpoint_path(output_root, scale, seed),
        context=context,
        scale=scale,
        seed=seed,
        launch_nonce=launch_nonce,
        trainer_sha256=trainer_sha256,
        transcript_root=transcript_root,
        execution_environment_binding=frozen_environment,
        return_raw=return_raw_checkpoint,
    )


def _direct_training_binding(
    context: FrozenContext,
    *,
    launch_nonce: str,
    trainer_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest": context.manifest_binding,
        "source_commit": context.source["commit"],
        "attestation": {
            "scheme": attestation.SCHEME,
            "key_id": context.manifest_binding["attestation"]["key_id"],
            "launch_nonce": launch_nonce,
        },
        "canonical_trainer_sha256": trainer_sha256,
        "hyperparameters": FROZEN_HYPERPARAMETERS,
        "seed_rules": SEED_RULES,
    }


def validate_training_summary(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    train_script: Path,
    context: FrozenContext,
    scale: str,
    seed: int,
    trust_root: attestation.TrustRoot,
    launch_nonce: str,
    trainer_sha256: str,
    expected_execution_environment: Mapping[str, Any] | None = None,
    return_raw_checkpoint: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], Mapping[str, Any]]:
    _require(
        trust_root.key_id == context.manifest_binding["attestation"]["key_id"],
        "Training attestation trust root does not match the manifest.",
    )
    _validate_payload_digest(payload, label="Training summary")
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Training summary attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=SUMMARY_ATTESTATION_PURPOSE,
    )
    _require(
        payload.get("direct_training_contract")
        == _direct_training_binding(
            context,
            launch_nonce=launch_nonce,
            trainer_sha256=trainer_sha256,
        ),
        "Training manifest/implementation binding drifted.",
    )
    return _validate_training_base(
        payload,
        output_root=output_root,
        train_script=train_script,
        context=context,
        scale=scale,
        seed=seed,
        launch_nonce=launch_nonce,
        trainer_sha256=trainer_sha256,
        expected_execution_environment=expected_execution_environment,
        return_raw_checkpoint=return_raw_checkpoint,
    )


def load_validated_training_summary(
    summary_path: Path,
    *,
    output_root: Path,
    train_script: Path,
    context: FrozenContext,
    scale: str,
    seed: int,
    trust_root: attestation.TrustRoot,
    launch_nonce: str,
    trainer_sha256: str,
    expected_execution_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _load_json(summary_path)
    _require(
        "payload_sha256" in payload
        and "direct_training_contract" in payload
        and "attestation" in payload,
        "Refusing an unauthenticated or partially bound direct training summary.",
    )
    validate_training_summary(
        payload,
        output_root=output_root,
        train_script=train_script,
        context=context,
        scale=scale,
        seed=seed,
        trust_root=trust_root,
        launch_nonce=launch_nonce,
        trainer_sha256=trainer_sha256,
        expected_execution_environment=expected_execution_environment,
    )
    return payload


def load_validated_training_bundle(
    summary_path: Path,
    *,
    output_root: Path,
    train_script: Path,
    context: FrozenContext,
    scale: str,
    seed: int,
    trust_root: attestation.TrustRoot,
    launch_nonce: str,
    trainer_sha256: str,
    expected_execution_environment: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    payload = _load_json(summary_path)
    validated = validate_training_summary(
        payload,
        output_root=output_root,
        train_script=train_script,
        context=context,
        scale=scale,
        seed=seed,
        trust_root=trust_root,
        launch_nonce=launch_nonce,
        trainer_sha256=trainer_sha256,
        expected_execution_environment=expected_execution_environment,
        return_raw_checkpoint=True,
    )
    _require(isinstance(validated, tuple), "Validated training bundle lost checkpoint bytes.")
    metadata, raw = cast(tuple[dict[str, Any], Mapping[str, Any]], validated)
    return payload, metadata, raw


def _run_record(
    payload: Mapping[str, Any],
    *,
    summary_path: Path,
    scale: str,
    seed: int,
    launch_nonce: str,
    trainer_sha256: str,
) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(summary_path)
    try:
        summary_sha256 = opened.sha256
        summary_bytes = opened.bytes
        opened.assert_unchanged()
    finally:
        opened.close()
    return {
        "scale": scale,
        "seed": seed,
        **_seed_values(seed),
        "steps_completed": STEPS,
        "launch_nonce": launch_nonce,
        "canonical_trainer_sha256": trainer_sha256,
        "checkpoint": payload["checkpoint"],
        "summary": {
            "path": str(summary_path),
            "sha256": summary_sha256,
            "bytes": summary_bytes,
            "payload_sha256": payload["payload_sha256"],
            "attestation_mac": payload["attestation"]["mac"],
        },
    }


def _matrix_payload(
    runs: Sequence[Mapping[str, Any]],
    *,
    context: FrozenContext,
    trust_root: attestation.TrustRoot,
    trainer_binding: Mapping[str, Any],
    gpu_lease_binding: Mapping[str, Any],
    execution_environment_binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected_runs = len(FROZEN_SCALES) * len(FROZEN_TRAINING_SEEDS)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal" if len(runs) == expected_runs else "in_progress",
        "source": context.source,
        "manifest": context.manifest_binding,
        "attestation_contract": context.manifest_binding["attestation"],
        "canonical_trainer": dict(trainer_binding),
        "gpu_lease": dict(gpu_lease_binding),
        "execution_environment": execution_environment.validate_execution_environment(
            execution_environment_binding
        ),
        "scales": list(FROZEN_SCALES),
        "frozen_training_seeds": list(FROZEN_TRAINING_SEEDS),
        "steps": STEPS,
        "minimum_steps": MINIMUM_STEPS,
        "seed_rules": SEED_RULES,
        "expected_runs": expected_runs,
        "completed_runs": len(runs),
        "runs": list(runs),
    }
    return _attested_payload(
        payload,
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
    )


def validate_matrix_summary(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    train_script: Path,
    context: FrozenContext,
    trust_root: attestation.TrustRoot,
    trainer_binding: Mapping[str, Any],
    expected_gpu_lease: Mapping[str, Any] | None = None,
    expected_execution_environment: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    _require(
        set(payload) == TRAINING_MATRIX_FIELDS,
        "Training matrix top-level schema drifted.",
    )
    _validate_payload_digest(payload, label="Training matrix")
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Training matrix attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
    )
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Training matrix schema drifted.")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong training matrix ID.")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "Training matrix type drifted.")
    _require(payload.get("source") == context.source, "Training matrix source drifted.")
    _require(
        payload.get("manifest") == context.manifest_binding, "Training matrix manifest drifted."
    )
    _require(
        payload.get("attestation_contract") == context.manifest_binding["attestation"],
        "Training matrix attestation contract drifted.",
    )
    _require(
        payload.get("canonical_trainer") == dict(trainer_binding),
        "Training matrix canonical trainer binding drifted.",
    )
    validated_gpu_lease = _validate_gpu_lease_binding(
        payload.get("gpu_lease"),
        expected=expected_gpu_lease,
    )
    raw_environment = payload.get("execution_environment")
    _require(
        isinstance(raw_environment, Mapping),
        "Training matrix execution environment is missing.",
    )
    frozen_environment = execution_environment.validate_execution_environment(
        cast(Mapping[str, Any], raw_environment)
    )
    _require(
        validated_gpu_lease["selected_device_class"]
        == execution_environment.selected_device_class(frozen_environment)
        and validated_gpu_lease["selected_device_routing_identity"]
        == execution_environment.selected_device_routing_identity(frozen_environment),
        "Training matrix GPU lease is not bound to its exact selected device.",
    )
    if expected_execution_environment is not None:
        _require(
            frozen_environment
            == execution_environment.validate_execution_environment(expected_execution_environment),
            "Training matrix execution environment changed on exact resume.",
        )
    _require(tuple(payload.get("scales", ())) == FROZEN_SCALES, "Training scales drifted.")
    _require(
        tuple(payload.get("frozen_training_seeds", ())) == FROZEN_TRAINING_SEEDS,
        "Training seed grid drifted.",
    )
    _require(payload.get("steps") == STEPS, "Training matrix step count drifted.")
    _require(payload.get("minimum_steps") == MINIMUM_STEPS, "Minimum steps drifted.")
    _require(payload.get("seed_rules") == SEED_RULES, "Matrix seed rules drifted.")
    coordinates = [(scale, seed) for scale in FROZEN_SCALES for seed in FROZEN_TRAINING_SEEDS]
    _require(payload.get("expected_runs") == len(coordinates), "Expected run count drifted.")
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("Training matrix run inventory is invalid.")
    _require(payload.get("completed_runs") == len(runs), "Completed run count drifted.")
    _require(len(runs) <= len(coordinates), "Training matrix contains extra runs.")
    expected_status = "terminal" if len(runs) == len(coordinates) else "in_progress"
    _require(payload.get("status") == expected_status, "Training matrix status drifted.")

    validated: list[dict[str, Any]] = []
    observed_nonces: set[str] = set()
    for record, (scale, seed) in zip(runs, coordinates[: len(runs)], strict=True):
        _require(isinstance(record, dict), "Training matrix run record is invalid.")
        launch_nonce = record.get("launch_nonce")
        trainer_sha256 = record.get("canonical_trainer_sha256")
        _require(
            contract.is_sha256(launch_nonce) and launch_nonce not in observed_nonces,
            "Training matrix launch nonce is invalid or replayed.",
        )
        observed_nonces.add(str(launch_nonce))
        _require(
            trainer_sha256 == trainer_binding["sha256"],
            "Training matrix trainer checksum drifted.",
        )
        summary_path = _training_summary_path(output_root, scale, seed)
        _require(summary_path.is_file(), "Training matrix references a missing summary.")
        summary = load_validated_training_summary(
            summary_path,
            output_root=output_root,
            train_script=train_script,
            context=context,
            scale=scale,
            seed=seed,
            trust_root=trust_root,
            launch_nonce=str(launch_nonce),
            trainer_sha256=str(trainer_sha256),
            expected_execution_environment=frozen_environment,
        )
        expected_record = _run_record(
            summary,
            summary_path=summary_path,
            scale=scale,
            seed=seed,
            launch_nonce=str(launch_nonce),
            trainer_sha256=str(trainer_sha256),
        )
        _require(record == expected_record, f"Training matrix run record drifted: {scale}/{seed}.")
        validated.append(expected_record)
    matrix_summary_path = Path(os.path.abspath(output_root)) / MATRIX_SUMMARY.name
    on_disk = _load_json(matrix_summary_path)
    _require(
        attestation.canonical_json(on_disk) == attestation.canonical_json(dict(payload)),
        "Training matrix ledger bytes do not match the supplied payload.",
    )
    _preflight_all_coordinates(
        output_root=output_root,
        matrix_summary=matrix_summary_path,
        completed_count=len(validated),
    )
    return validated


def load_terminal_matrix_record(
    matrix_summary: Path,
    *,
    context: FrozenContext,
    trust_root: attestation.TrustRoot,
    trainer_binding: Mapping[str, Any],
    scale: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _load_json(matrix_summary)
    validated_records = validate_matrix_summary(
        payload,
        output_root=matrix_summary.parent,
        train_script=TRAIN_SCRIPT,
        context=context,
        trust_root=trust_root,
        trainer_binding=trainer_binding,
    )
    _validate_payload_digest(payload, label="Training matrix")
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Training matrix attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
    )
    _require(
        payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("artifact_type") == ARTIFACT_TYPE
        and payload.get("status") == "terminal"
        and payload.get("completed_runs") == payload.get("expected_runs") == 10,
        "Terminal training matrix ledger is incomplete.",
    )
    _require(payload.get("source") == context.source, "Training ledger source drifted.")
    _require(
        payload.get("manifest") == context.manifest_binding, "Training ledger manifest drifted."
    )
    _require(
        payload.get("canonical_trainer") == dict(trainer_binding),
        "Training ledger canonical trainer drifted.",
    )
    _validate_gpu_lease_binding(payload.get("gpu_lease"))
    runs = validated_records
    _require(len(runs) == 10, "Training ledger run grid is invalid.")
    coordinates = [
        (item_scale, item_seed)
        for item_scale in FROZEN_SCALES
        for item_seed in FROZEN_TRAINING_SEEDS
    ]
    observed_nonces: set[str] = set()
    selected: dict[str, Any] | None = None
    for record, coordinate in zip(runs, coordinates, strict=True):
        _require(isinstance(record, dict), "Training ledger record is invalid.")
        _require(
            (record.get("scale"), record.get("seed")) == coordinate,
            "Training ledger coordinate order drifted.",
        )
        nonce = record.get("launch_nonce")
        _require(
            contract.is_sha256(nonce) and nonce not in observed_nonces,
            "Training ledger nonce is invalid or replayed.",
        )
        observed_nonces.add(str(nonce))
        _require(
            record.get("canonical_trainer_sha256") == trainer_binding["sha256"],
            "Training ledger trainer checksum drifted.",
        )
        if coordinate == (scale, seed):
            selected = dict(record)
    _require(selected is not None, "Requested training coordinate is absent from terminal ledger.")
    return payload, cast(dict[str, Any], selected)


def _assert_no_orphaned_output(output_dir: Path) -> None:
    if output_dir.exists():
        _require(
            not any(output_dir.iterdir()),
            f"Refusing to overwrite orphaned or stale training output: {output_dir}",
        )


@contextmanager
def _matrix_lock(output_root: Path) -> Iterator[_MatrixLockLease]:
    lock_path = Path(os.path.abspath(output_root)).parent / LOCK_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Training matrix locking requires O_NOFOLLOW support.")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise ValueError("Training matrix lock could not be opened safely.") from error
    acquired = False
    try:
        opened = os.fstat(descriptor)
        lease = _MatrixLockLease(
            path=lock_path,
            file_descriptor=descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
        )
        lease.assert_held()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another direct-training matrix runner holds the exclusive lock."
            ) from error
        acquired = True
        lease.assert_held()
        try:
            yield lease
        finally:
            lease.assert_held()
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def _cell_claim(output_dir: Path, *, launch_nonce: str) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    claim = output_dir / CLAIM_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(claim, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(
            f"Direct-training cell already has a launch claim: {output_dir}"
        ) from error
    completed = False
    try:
        payload = (launch_nonce + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
        completed = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if completed:
            claim.unlink()
            directory_fd = os.open(output_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)


def _preflight_all_coordinates(
    *,
    output_root: Path,
    matrix_summary: Path,
    completed_count: int,
) -> None:
    root = Path(os.path.abspath(output_root))
    summary = Path(os.path.abspath(matrix_summary))
    _require(
        summary == root / MATRIX_SUMMARY.name,
        "Training matrix summary must use its canonical output-root path.",
    )
    if not root.exists():
        _require(completed_count == 0, "Completed training prefix lost its output root.")
        return
    _require(root.is_dir() and not root.is_symlink(), "Training output root is unsafe.")
    allowed: set[Path] = {summary}
    coordinates = [(scale, seed) for scale in FROZEN_SCALES for seed in FROZEN_TRAINING_SEEDS]
    for index, (scale, seed) in enumerate(coordinates):
        scale_dir = root / scale
        output_dir = _run_output_dir(root, scale, seed)
        summary_path = _training_summary_path(root, scale, seed)
        checkpoint_path = _checkpoint_path(root, scale, seed)
        claim_path = output_dir / CLAIM_NAME
        allowed.update((scale_dir, output_dir))
        _require(
            not os.path.lexists(claim_path),
            f"Crash-stale training claim detected: {scale}/{seed}.",
        )
        if index < completed_count:
            allowed.update((summary_path, checkpoint_path))
            _require(
                output_dir.is_dir()
                and not output_dir.is_symlink()
                and summary_path.is_file()
                and not summary_path.is_symlink()
                and checkpoint_path.is_file()
                and not checkpoint_path.is_symlink(),
                f"Completed training bundle disappeared or became unsafe: {scale}/{seed}.",
            )
            continue
        if output_dir.exists():
            _require(
                output_dir.is_dir()
                and not output_dir.is_symlink()
                and not any(output_dir.iterdir()),
                f"Refusing orphaned or stale future training output: {scale}/{seed}.",
            )
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"Training output tree contains a symlink: {item}")
        _require(
            Path(os.path.abspath(item)) in allowed,
            f"Training output tree contains an unregistered orphan: {item}",
        )


def _run_matrix_under_gpu_lease(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path = MATRIX_SUMMARY,
    train_script: Path = TRAIN_SCRIPT,
    steps: int = STEPS,
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
        "Training GPU lease binding does not match the held lease.",
    )
    gpu_lease.assert_held()
    device_guard.assert_held()
    _require(tuple(contract.SCALES) == FROZEN_SCALES, "Contract training scales drifted.")
    _require(
        tuple(contract.TRAINING_SEEDS) == FROZEN_TRAINING_SEEDS,
        "Contract training seeds drifted.",
    )
    _require(steps == STEPS, f"Direct training requires exactly {STEPS} steps.")
    canonical_trainer = _canonical_train_script(train_script)
    _require(not output_root.is_symlink(), "Training output root may not be a symbolic link.")
    _require(not matrix_summary.is_symlink(), "Training matrix may not be a symbolic link.")
    _require(
        Path(os.path.abspath(matrix_summary))
        == Path(os.path.abspath(output_root)) / MATRIX_SUMMARY.name,
        "Training matrix summary must use its canonical output-root path.",
    )
    context = establish_frozen_context(manifest_path)
    expected_key_id = str(context.manifest_binding["attestation"]["key_id"])
    if attestation_key_path is None:
        trust_root = attestation.trust_root_from_environment(
            repository_root=REPOSITORY_ROOT,
            artifact_roots=(output_root,),
            expected_key_id=expected_key_id,
        )
    else:
        trust_root = attestation.load_trust_root(
            attestation_key_path,
            repository_root=REPOSITORY_ROOT,
            artifact_roots=(output_root,),
            expected_key_id=expected_key_id,
        )
    opened_trainer, trainer_snapshot = _open_canonical_trainer(canonical_trainer)
    try:
        opened_trainer.assert_unchanged()
    finally:
        opened_trainer.close()
    _require(
        contract.implementation_tree_digest() == context.manifest_binding["implementation_digest"],
        "Canonical trainer is not bound by the frozen implementation manifest.",
    )
    trainer_binding = trainer_snapshot.public_binding

    with _matrix_lock(output_root) as matrix_lock:
        matrix_lock.assert_held()
        gpu_lease.assert_held()
        device_guard.assert_held()
        completed: list[dict[str, Any]] = []
        if matrix_summary.exists():
            gpu_lease.assert_held()
            device_guard.assert_held()
            completed = validate_matrix_summary(
                _load_json(matrix_summary),
                output_root=output_root,
                train_script=canonical_trainer,
                context=context,
                trust_root=trust_root,
                trainer_binding=trainer_binding,
                expected_gpu_lease=gpu_lease_binding,
                expected_execution_environment=frozen_execution_environment,
            )
            matrix_lock.assert_held()
        _preflight_all_coordinates(
            output_root=output_root,
            matrix_summary=matrix_summary,
            completed_count=len(completed),
        )
        assert_environment_unchanged(context)
        verification, _ = _open_canonical_trainer(canonical_trainer, expected=trainer_snapshot)
        verification.close()
        matrix_lock.assert_held()
        if not matrix_summary.exists():
            gpu_lease.assert_held()
            device_guard.assert_held()
            _publish_matrix_ledger(
                matrix_summary,
                _matrix_payload(
                    completed,
                    context=context,
                    trust_root=trust_root,
                    trainer_binding=trainer_binding,
                    gpu_lease_binding=gpu_lease_binding,
                    execution_environment_binding=frozen_execution_environment,
                ),
                matrix_lock=matrix_lock,
            )
            gpu_lease.assert_held()
            device_guard.assert_held()

        coordinates = [(scale, seed) for scale in FROZEN_SCALES for seed in FROZEN_TRAINING_SEEDS]
        for scale, seed in coordinates[len(completed) :]:
            matrix_lock.assert_held()
            gpu_lease.assert_held()
            device_guard.assert_held()
            assert_environment_unchanged(context)
            execution_environment.assert_exact_execution_environment(frozen_execution_environment)
            gpu_lease.assert_held()
            device_guard.assert_held()
            launch_nonce = secrets.token_hex(32)
            summary_path = _training_summary_path(output_root, scale, seed)
            output_dir = _run_output_dir(output_root, scale, seed)
            with _cell_claim(output_dir, launch_nonce=launch_nonce):
                command = build_training_command(
                    train_script=canonical_trainer,
                    output_root=output_root,
                    scale=scale,
                    seed=seed,
                    context=context,
                    launch_nonce=launch_nonce,
                    trainer_sha256=trainer_snapshot.sha256,
                    device_index=cast(int, frozen_execution_environment["current_device_index"]),
                    device_routing_identity=(
                        execution_environment.selected_device_routing_identity(
                            frozen_execution_environment
                        )
                    ),
                )
                gpu_lease.assert_held()
                device_guard.assert_held()
                result = _run_trainer_from_stable_script(
                    command,
                    canonical=canonical_trainer,
                    expected=trainer_snapshot,
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
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, command)
                assert_environment_unchanged(context)
                _require(
                    summary_path.is_file(),
                    f"Training did not publish its summary: {scale}/{seed}.",
                )
                summary = load_validated_training_summary(
                    summary_path,
                    output_root=output_root,
                    train_script=canonical_trainer,
                    context=context,
                    scale=scale,
                    seed=seed,
                    trust_root=trust_root,
                    launch_nonce=launch_nonce,
                    trainer_sha256=trainer_snapshot.sha256,
                    expected_execution_environment=frozen_execution_environment,
                )
                matrix_lock.assert_held()
                record = _run_record(
                    summary,
                    summary_path=summary_path,
                    scale=scale,
                    seed=seed,
                    launch_nonce=launch_nonce,
                    trainer_sha256=trainer_snapshot.sha256,
                )
                matrix_lock.assert_held()
                completed.append(record)
                gpu_lease.assert_held()
                device_guard.assert_held()
                _publish_matrix_ledger(
                    matrix_summary,
                    _matrix_payload(
                        completed,
                        context=context,
                        trust_root=trust_root,
                        trainer_binding=trainer_binding,
                        gpu_lease_binding=gpu_lease_binding,
                        execution_environment_binding=frozen_execution_environment,
                    ),
                    matrix_lock=matrix_lock,
                )
                gpu_lease.assert_held()

        gpu_lease.assert_held()
        device_guard.assert_held()
        assert_environment_unchanged(context)
        terminal = _matrix_payload(
            completed,
            context=context,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
            gpu_lease_binding=gpu_lease_binding,
            execution_environment_binding=frozen_execution_environment,
        )
        gpu_lease.assert_held()
        device_guard.assert_held()
        _publish_matrix_ledger(matrix_summary, terminal, matrix_lock=matrix_lock)
        gpu_lease.assert_held()
        device_guard.assert_held()
        validate_matrix_summary(
            terminal,
            output_root=output_root,
            train_script=canonical_trainer,
            context=context,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
            expected_gpu_lease=gpu_lease_binding,
            expected_execution_environment=frozen_execution_environment,
        )
        matrix_lock.assert_held()
        gpu_lease.assert_held()
        device_guard.assert_held()
        return terminal


def run_matrix(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path = MATRIX_SUMMARY,
    train_script: Path = TRAIN_SCRIPT,
    steps: int = STEPS,
    attestation_key_path: Path | None = None,
    gpu_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Run the complete CUDA training matrix under one project-wide GPU lease."""

    requested_gpu_lock = gpu_lock.DEFAULT_LOCK_PATH if gpu_lock_path is None else gpu_lock_path
    lease = gpu_lock.acquire_gpu_lock(
        "p2-direct-training-matrix",
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
            "p2-direct-training-matrix",
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
            output_root=output_root,
            matrix_summary=matrix_summary,
            train_script=train_script,
            steps=steps,
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
        description="Resume-safe launcher for the frozen post-rank direct-controller training grid."
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--matrix-summary", type=Path, default=MATRIX_SUMMARY)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--gpu-lock-path", type=Path)
    args = parser.parse_args()
    result = run_matrix(
        manifest_path=args.manifest,
        output_root=args.output_root,
        matrix_summary=args.matrix_summary,
        steps=args.steps,
        gpu_lock_path=args.gpu_lock_path,
    )
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "completed_runs": result["completed_runs"],
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
