from __future__ import annotations

import argparse
import fcntl
import importlib
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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import adaptive_v4_execution_environment as execution_environment
import calibrate_p2_direct_soft_lag as calibration
import p2_direct_attestation as attestation
import p2_direct_controller_contract as contract
import run_p2_direct_calibration_matrix as calibration_matrix
import run_p2_direct_top_p_physical_matrix as top_p_matrix
import run_p2_direct_training_matrix as training_matrix
from adaptive_v4_gpu_lock import (
    DEVICE_GUARD_SEMANTICS,
    GPULockLease,
    acquire_device_guard,
    acquire_gpu_lock,
    canonical_device_guard_path,
)
from adaptive_v4_gpu_lock import SAFE_LOCK_MODE as GPU_LOCK_MODE

DEFAULT_GPU_LOCK_PATH = contract.DIRECT_GPU_SCHEDULER_LOCK_PATH

EXPERIMENT_ID = "p2-post-rank-direct-controller-matrix-v1"
ARTIFACT_TYPE = "direct-controller-quality-matrix"
SCHEMA_VERSION = 1
MATRIX_ATTESTATION_PURPOSE = "p2-direct-controller-matrix-v1"
WORKER_ATTESTATION_PURPOSE = "p2-direct-controller-worker-ledger-v1"
WORKER_ARTIFACT_TYPE = "direct-controller-worker-ledger"
WORKER_ASSIGNMENT_RULE = "canonical-coordinate-index-modulo-worker-count-v1"
GPU_LEASE_SEMANTICS = "device-scoped-nonblocking-persistent-inode-worker-lease-v1"
GPU_LOCK_IMPLEMENTATION_PATH = "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py"
DEVICE_GUARD_PATH_DERIVATION = "fixed-tmp-root-full-sha256-of-identity-type-nul-identity-v1"

FROZEN_SCALES = tuple(contract.SCALES)
FROZEN_TRAINING_SEEDS = tuple(contract.TRAINING_SEEDS)
FROZEN_BUDGETS = tuple(contract.BUDGETS)
FROZEN_FAMILIES = tuple(contract.FAMILIES)
FROZEN_CONTEXTS = tuple(contract.CONTEXTS)
FROZEN_REPLICATES = tuple(contract.REPLICATES)
EXAMPLES_PER_SHARD = contract.EXAMPLES_PER_SHARD
ARM_NAMES = tuple(contract.ALL_ARM_NAMES)
EXPECTED_SHARDS = (
    len(FROZEN_SCALES)
    * len(FROZEN_TRAINING_SEEDS)
    * len(FROZEN_BUDGETS)
    * len(FROZEN_FAMILIES)
    * len(FROZEN_CONTEXTS)
    * len(FROZEN_REPLICATES)
)
EXPECTED_OUTCOME_ROWS = EXPECTED_SHARDS * EXAMPLES_PER_SHARD * len(ARM_NAMES)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR_SCRIPT = Path(__file__).resolve().with_name("evaluate_p2_direct_controller_shard.py")
EVALUATOR_IMPLEMENTATION_PATH = (
    "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard.py"
)
EVALUATOR_FD_ENV = "ADAPTIVE_V4_CANONICAL_DIRECT_EVALUATOR_FD"
STORAGE_PROJECTED_SHARDS_ENV = "ADAPTIVE_V4_DIRECT_PROJECTED_REMAINING_SHARDS"
STORAGE_PROJECTED_TOKEN_ROWS_ENV = "ADAPTIVE_V4_DIRECT_PROJECTED_REMAINING_TOKEN_ROWS"
EVALUATOR_FD_BOOTSTRAP = (
    "import os,sys;"
    "p=sys.argv.pop(1);"
    "sys.path.insert(0,os.path.dirname(p));"
    f"fd=int(os.environ[{EVALUATOR_FD_ENV!r}]);"
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
TOP_P_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/top_p_physical_match"
)
OUTPUT_ROOT = Path("artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller")
MATRIX_SUMMARY_NAME = "controller-matrix.summary.json"
MATRIX_SUMMARY = OUTPUT_ROOT / MATRIX_SUMMARY_NAME
MATRIX_LOCK_SUFFIX = "p2-direct-controller-matrix.lock"
MATRIX_LOCK_SEMANTICS = "persistent-sibling-flock-exclusive-process-owner-v1"
CELL_CLAIM_NAME = ".p2-direct-controller-cell.claim"
CELL_CLAIM_SEMANTICS = "exclusive-create-coordinate-nonce-claim-v1"
DEVICE = "cuda"
DTYPE = "bfloat16"
BUNDLE_KINDS = ("examples", "outcomes", "tokens", "failures")
CRASH_RECOVERY_BOUNDARY = (
    "fail-closed: a claim or any evaluator bundle member outside the MAC-attested completed "
    "matrix prefix is orphan evidence and is never auto-promoted; quarantine is required"
)
OUTCOME_POLICY = (
    "execute every frozen coordinate regardless of raw correctness or prior shard contents; "
    "only structural integrity controls publication and resume"
)
FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES = 1 << 30
NEXT_LAUNCH_HEADROOM_METHOD = "observed-component-high-water-next-coordinate-v1"
NEXT_LAUNCH_HEADROOM_LIMITATIONS = (
    "planning headroom only: observed compressed sizes can understate an unseen shard; "
    "the check does not reserve filesystem space, exclude concurrent writers, or guarantee "
    "that the next bundle write will complete"
)
WORKER_LEDGER_ROOT_SEMANTICS = "closed-world-persistent-sibling-worker-ledger-root-v1"
WORKER_LEDGER_PATH_SEMANTICS = "canonical-worker-index-and-count-filename-v1"
INFRASTRUCTURE_PAUSE_RESUME_RULE = (
    "revalidate-attested-pause-and-bound-records; rerun-frozen-headroom-for-the-same-next-"
    "coordinate; transition-to-in_progress-before-claim-only-when-sufficient-v1"
)
INFRASTRUCTURE_DRAIN_RULE = (
    "block-all-new-claims; allow-only-preexisting-claims-to-commit; reattest-latest-sparse-"
    "projection; only-bound-owner-may-rerun-headroom-and-resume-v1"
)
_HEADROOM_SIGNAL_FIELDS = frozenset(
    {
        "schema_version",
        "signal_type",
        "retryable",
        "coordinate_key",
        "method",
        "outcome_independent",
        "completed_record_count",
        "positive_token_rate_observation_count",
        "next_coordinate_decode_token_rows",
        "observed_next_shard_estimate_compressed_bytes",
        "frozen_minimum_one_shard_headroom_bytes",
        "required_headroom_bytes",
        "filesystem_available_bytes",
        "sufficient",
        "limitations",
    }
)
_INFRASTRUCTURE_PAUSE_FIELDS = frozenset(
    {
        *_HEADROOM_SIGNAL_FIELDS,
        "resume_rule",
        "matrix_records_digest",
        "canonical_prefix_shards",
        "globally_committed_shards",
        "worker_ledger_registry_digest",
        "resume_matrix_payload_sha256",
        "pause_worker_index",
        "pause_worker_count",
    }
)
_INFRASTRUCTURE_DRAIN_FIELDS = frozenset(
    {
        *_INFRASTRUCTURE_PAUSE_FIELDS,
        "drain_rule",
        "active_claim_coordinate_indices",
        "active_claim_count",
    }
)


class InfrastructurePauseSignal(RuntimeError):
    """Typed, retryable pause raised before a shard launch when headroom is insufficient."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        state = "insufficient filesystem headroom"
        if self.payload.get("sufficient") is True:
            state = "in-flight infrastructure drain awaiting its bound owner"
        super().__init__(
            f"Direct-controller launch blocked for {state}: "
            f"required={self.payload['required_headroom_bytes']} "
            f"available={self.payload['filesystem_available_bytes']}"
        )


_WORKER_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "source",
        "manifest",
        "attestation_contract",
        "prerequisites",
        "canonical_evaluator",
        "matrix_lock",
        "gpu_lease",
        "worker_index",
        "worker_count",
        "assignment_rule",
        "global_coordinate_count",
        "global_coordinate_digest",
        "assigned_coordinate_count",
        "assigned_coordinate_digest",
        "completed_shards",
        "integrity_pass_shards",
        "integrity_fail_shards",
        "outcome_dependent_early_stopping",
        "outcome_selection_performed",
        "quality_outcomes_aggregated",
        "records",
        "payload_sha256",
        "attestation",
    }
)

_MATRIX_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "artifact_type",
        "status",
        "integrity_status",
        "source",
        "manifest",
        "attestation_contract",
        "prerequisites",
        "canonical_evaluator",
        "matrix_lock",
        "gpu_worker_leases",
        "gpu_device_identity_registry",
        "selected_device_class",
        "worker_ledger_assignment_rule",
        "worker_ledger_root",
        "worker_ledger_registry",
        "execution_mode",
        "worker_count",
        "cell_claim_semantics",
        "crash_recovery_boundary",
        "outcome_policy",
        "outcome_dependent_early_stopping",
        "outcome_selection_performed",
        "quality_outcomes_aggregated",
        "coordinate_order",
        "coordinate_count",
        "coordinate_digest",
        "examples_per_shard",
        "arm_names",
        "expected_outcome_rows",
        "completed_shards",
        "canonical_prefix_shards",
        "globally_committed_shards",
        "integrity_pass_shards",
        "integrity_fail_shards",
        "storage_aggregation_scope",
        "storage",
        "records",
        "payload_sha256",
        "attestation",
    }
)

_PAUSED_MATRIX_FIELDS = frozenset(
    (_MATRIX_FIELDS - {"integrity_status", "integrity_pass_shards", "integrity_fail_shards"})
    | {"infrastructure_pause"}
)
_DRAINING_MATRIX_FIELDS = frozenset(
    (_MATRIX_FIELDS - {"integrity_status", "integrity_pass_shards", "integrity_fail_shards"})
    | {"infrastructure_drain"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ShardCoordinate:
    scale: str
    training_seed: int
    calibration_seed: int
    evaluation_seed: int
    budget: str
    family: str
    context: int
    replicate: int
    generation_seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.scale}/train-{self.training_seed}/{self.budget}/{self.family}/"
            f"context-{self.context}/replicate-{self.replicate}"
        )

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "training_seed": self.training_seed,
            "calibration_seed": self.calibration_seed,
            "evaluation_seed": self.evaluation_seed,
            "budget": self.budget,
            "family": self.family,
            "context": self.context,
            "replicate": self.replicate,
            "generation_seed": self.generation_seed,
        }

    @property
    def evaluator_payload(self) -> dict[str, Any]:
        return {
            **self.payload,
            "global_block_budget": contract.DIRECT_GLOBAL_BLOCK_BUDGETS[self.scale][self.budget],
            "csa_layers": list(contract.DIRECT_CSA_LAYERS_BY_SCALE[self.scale]),
        }


@dataclass(frozen=True)
class CanonicalEvaluatorSnapshot:
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
            "implementation_path": EVALUATOR_IMPLEMENTATION_PATH,
        }


@dataclass(frozen=True)
class MatrixLayout:
    output_root: Path
    matrix_summary: Path
    lock_path: Path
    training_output_root: Path
    calibration_output_root: Path
    top_p_output_root: Path


def _worker_ledger_root(output_root: Path) -> Path:
    root = _absolute(output_root)
    return root.parent / f".{root.name}.p2-direct-controller-workers"


def _worker_ledger_path(output_root: Path, *, worker_index: int, worker_count: int) -> Path:
    return _worker_ledger_root(output_root) / (
        f"worker-{worker_index:05d}-of-{worker_count:05d}.summary.json"
    )


@dataclass(frozen=True)
class InputBundle:
    coordinate: tuple[str, int, str]
    checkpoint_path: Path
    training_summary_path: Path
    training_matrix_summary_path: Path
    calibration_path: Path
    top_p_05_path: Path
    top_p_08_path: Path
    calibration_payload: Mapping[str, Any]
    binding: Mapping[str, Any]


@dataclass(frozen=True)
class FrozenPrerequisites:
    context: training_matrix.FrozenContext
    trust_root: attestation.TrustRoot
    bundles: Mapping[tuple[str, int, str], InputBundle]
    public_binding: Mapping[str, Any]


_ACTIVE_MATRIX_LOCKS: set[str] = set()
_ACTIVE_MATRIX_LOCKS_GUARD = threading.Lock()
_MATRIX_LOCK_GUARDIANS: dict[str, int] = {}
_MATRIX_LOCK_GUARDIANS_GUARD = threading.Lock()


def _process_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        value = int(fields[19])
    except (OSError, ValueError, IndexError) as error:
        raise ValueError(f"Cannot bind controller claim to process {pid}.") from error
    _require(value > 0, "Controller claim process start time is invalid.")
    return value


def coordinates() -> tuple[ShardCoordinate, ...]:
    result: list[ShardCoordinate] = []
    for scale in FROZEN_SCALES:
        for training_seed in FROZEN_TRAINING_SEEDS:
            aligned, calibration_seed, evaluation_seed = contract.seed_triplet(training_seed)
            _require(aligned == training_seed, "Frozen seed triplet alignment drifted.")
            for budget in FROZEN_BUDGETS:
                for family in FROZEN_FAMILIES:
                    for context in FROZEN_CONTEXTS:
                        for replicate in FROZEN_REPLICATES:
                            result.append(
                                ShardCoordinate(
                                    scale=scale,
                                    training_seed=training_seed,
                                    calibration_seed=calibration_seed,
                                    evaluation_seed=evaluation_seed,
                                    budget=budget,
                                    family=family,
                                    context=context,
                                    replicate=replicate,
                                    generation_seed=contract.generation_seed(
                                        evaluation_seed, family, context, replicate
                                    ),
                                )
                            )
    _require(len(result) == EXPECTED_SHARDS == 9_000, "Direct grid is not exactly 9,000 shards.")
    _require(len({item.key for item in result}) == len(result), "Direct grid repeats a shard.")
    return tuple(result)


def coordinate_digest() -> str:
    return contract.json_digest([item.payload for item in coordinates()])


def expected_decode_token_rows(coordinate: ShardCoordinate) -> int:
    return (
        contract.DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT[coordinate.family][coordinate.context]
        * EXAMPLES_PER_SHARD
        * len(ARM_NAMES)
    )


def projected_decode_token_rows(items: Sequence[ShardCoordinate]) -> int:
    total = sum(expected_decode_token_rows(item) for item in items)
    _require(total > 0, "Projected remaining coordinate token weight is empty.")
    return total


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


def _matrix_lock_path(output_root: Path) -> Path:
    root = _absolute(output_root)
    _require(root != root.parent and bool(root.name), "Controller output root may not be a root.")
    return root.parent / f".{root.name}.{MATRIX_LOCK_SUFFIX}"


def _canonical_shard_filename(coordinate: ShardCoordinate) -> str:
    return (
        f"direct-{coordinate.scale}-train-{coordinate.training_seed}-{coordinate.budget}-"
        f"{coordinate.family}-context-{coordinate.context}-replicate-{coordinate.replicate}.json"
    )


def shard_output_dir(output_root: Path, coordinate: ShardCoordinate) -> Path:
    return (
        output_root
        / coordinate.scale
        / f"seed-{coordinate.training_seed}"
        / coordinate.budget
        / coordinate.family
        / f"context-{coordinate.context}"
        / f"replicate-{coordinate.replicate}"
    )


def shard_envelope_path(output_root: Path, coordinate: ShardCoordinate) -> Path:
    return shard_output_dir(output_root, coordinate) / _canonical_shard_filename(coordinate)


def canonical_bundle_paths(output_root: Path, coordinate: ShardCoordinate) -> dict[str, Path]:
    envelope = shard_envelope_path(output_root, coordinate)
    return {
        "envelope": envelope,
        **{kind: envelope.with_name(f"{envelope.stem}.{kind}.jsonl.gz") for kind in BUNDLE_KINDS},
    }


def _validate_matrix_layout(
    *,
    output_root: Path,
    matrix_summary: Path,
    training_output_root: Path,
    calibration_output_root: Path,
    top_p_output_root: Path,
    attestation_key_path: Path | None,
) -> MatrixLayout:
    root = _exact_resolved_path(output_root, label="Controller output root")
    summary = _exact_resolved_path(matrix_summary, label="Controller matrix summary")
    _require(
        summary == root / MATRIX_SUMMARY_NAME,
        "Controller matrix summary must use its canonical output-root path.",
    )
    input_roots = tuple(
        _exact_resolved_path(path, label=label)
        for path, label in (
            (training_output_root, "Training input root"),
            (calibration_output_root, "Calibration input root"),
            (top_p_output_root, "Top-p input root"),
        )
    )
    superseded_calibration_root = _absolute(
        calibration_matrix.SUPERSEDED_OUTPUT_ROOT
    ).resolve(strict=False)
    _require(
        all(
            not _paths_overlap(item, superseded_calibration_root)
            for item in (root, *input_roots, _worker_ledger_root(root))
        ),
        "Controller paths may not overlap the immutable revision 1.1 calibration quarantine.",
    )
    _require(
        all(
            not _paths_overlap(left, right)
            for i, left in enumerate(input_roots)
            for right in input_roots[i + 1 :]
        ),
        "Controller prerequisite roots must be pairwise disjoint.",
    )
    _require(
        all(not _paths_overlap(root, item) for item in input_roots),
        "Controller output root must be disjoint from every prerequisite root.",
    )
    lock_path = _exact_resolved_path(_matrix_lock_path(root), label="Controller matrix lock")
    _require(
        not _paths_overlap(lock_path, root), "Controller lock must be outside the output tree."
    )
    _require(
        all(not _paths_overlap(lock_path, item) for item in input_roots),
        "Controller lock may not overlap prerequisite roots.",
    )
    _require(
        not _paths_overlap(lock_path, superseded_calibration_root),
        "Controller lock may not overlap the immutable calibration quarantine.",
    )
    if attestation_key_path is not None:
        key = _exact_resolved_path(attestation_key_path, label="Attestation key")
        _require(
            not key.is_relative_to(REPOSITORY_ROOT.resolve(strict=True)),
            "Attestation key must be outside the repository.",
        )
        _require(
            not _paths_overlap(key, root)
            and key != lock_path
            and all(not _paths_overlap(key, item) for item in input_roots),
            "Attestation key must be disjoint from all controller paths.",
        )
    all_paths: set[Path] = set()
    for coordinate in coordinates():
        bundle = canonical_bundle_paths(root, coordinate)
        _require(
            not any(path in all_paths for path in bundle.values()),
            "Canonical evaluator bundle paths collide.",
        )
        all_paths.update(bundle.values())
        _require(summary not in bundle.values(), "Matrix summary collides with a shard bundle.")
    return MatrixLayout(
        output_root=root,
        matrix_summary=summary,
        lock_path=lock_path,
        training_output_root=input_roots[0],
        calibration_output_root=input_roots[1],
        top_p_output_root=input_roots[2],
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


def _opened_matrix_lock_binding(
    path: Path, *, create: bool, retain_process_guardian: bool = False
) -> dict[str, Any]:
    key = str(path)
    if retain_process_guardian:
        with _MATRIX_LOCK_GUARDIANS_GUARD:
            guarded = _MATRIX_LOCK_GUARDIANS.get(key)
            if guarded is not None:
                held = os.fstat(guarded)
                try:
                    current = os.stat(path, follow_symlinks=False)
                except FileNotFoundError as error:
                    raise ValueError("Persistent controller matrix lock was unlinked.") from error
                _require(
                    (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)
                    and held.st_nlink == 1,
                    "Persistent controller matrix lock was replaced.",
                )
                return {
                    "path": str(path),
                    "semantics": MATRIX_LOCK_SEMANTICS,
                    "persistent_inode": True,
                    "device": held.st_dev,
                    "inode": held.st_ino,
                    "unlink_on_release": False,
                }
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Controller locking requires O_NOFOLLOW.")
    flags = (
        (os.O_RDWR if create else os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
    )
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileNotFoundError as error:
        raise ValueError("Persistent controller matrix lock is missing.") from error
    keep_open = False
    try:
        opened = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
            and opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o600,
            "Controller lock identity or metadata is unsafe.",
        )
        binding = {
            "path": str(path),
            "semantics": MATRIX_LOCK_SEMANTICS,
            "persistent_inode": True,
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "unlink_on_release": False,
        }
        if retain_process_guardian:
            with _MATRIX_LOCK_GUARDIANS_GUARD:
                guarded = _MATRIX_LOCK_GUARDIANS.get(key)
                if guarded is None:
                    _MATRIX_LOCK_GUARDIANS[key] = descriptor
                    keep_open = True
                else:
                    held = os.fstat(guarded)
                    _require(
                        (held.st_dev, held.st_ino) == (opened.st_dev, opened.st_ino),
                        "Concurrent controller lock guardians disagree on the inode.",
                    )
                    binding["device"] = held.st_dev
                    binding["inode"] = held.st_ino
        return binding
    finally:
        if not keep_open:
            os.close(descriptor)


def _initialize_matrix_lock_binding(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return _opened_matrix_lock_binding(path, create=True, retain_process_guardian=True)


def _matrix_lock_binding(path: Path) -> dict[str, Any]:
    return _opened_matrix_lock_binding(path, create=False)


def _assert_lock_path_identity(lock_path: Path, descriptor: int, opened: os.stat_result) -> None:
    try:
        current = os.stat(lock_path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("Controller lock path was unlinked while held.") from error
    held = os.fstat(descriptor)
    _require(
        (held.st_dev, held.st_ino)
        == (opened.st_dev, opened.st_ino)
        == (current.st_dev, current.st_ino)
        and held.st_nlink == 1,
        "Controller lock path was replaced or unlinked while held.",
    )


@contextmanager
def _exclusive_matrix_lock(
    lock_path: Path,
    *,
    matrix_summary: Path,
    expected_binding: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    key = str(lock_path)
    with _ACTIVE_MATRIX_LOCKS_GUARD:
        _require(key not in _ACTIVE_MATRIX_LOCKS, "Controller matrix lock reentry is forbidden.")
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
        _require(no_follow is not None, "Controller locking requires O_NOFOLLOW.")
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
            0o600,
        )
        opened = os.fstat(descriptor)
        current = os.stat(lock_path, follow_symlinks=False)
        _require(stat.S_ISREG(opened.st_mode), "Controller lock is not regular.")
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Controller lock path changed while opening.",
        )
        _require(
            opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o600,
            "Controller lock ownership, links, or mode are unsafe.",
        )
        if expected_binding is not None:
            _require(
                expected_binding.get("path") == str(lock_path)
                and expected_binding.get("semantics") == MATRIX_LOCK_SEMANTICS
                and expected_binding.get("device") == opened.st_dev
                and expected_binding.get("inode") == opened.st_ino,
                "Controller lock inode differs from its persistent binding.",
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = True
        current = os.stat(lock_path, follow_symlinks=False)
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Controller lock path was replaced before ownership.",
        )
        _write_fd_json(descriptor, owner)
        yield
        _assert_lock_path_identity(lock_path, descriptor, opened)
        _write_fd_json(
            descriptor,
            {**owner, "state": "released", "released_time_ns": time.time_ns()},
        )
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        acquired = False
    finally:
        if descriptor is not None:
            if acquired:
                try:
                    _write_fd_json(
                        descriptor,
                        {
                            **owner,
                            "state": "released-after-error",
                            "released_time_ns": time.time_ns(),
                        },
                    )
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        with _ACTIVE_MATRIX_LOCKS_GUARD:
            _ACTIVE_MATRIX_LOCKS.discard(key)


def _assert_cell_claim_binding(binding: Mapping[str, Any]) -> None:
    path = Path(cast(str, binding.get("path")))
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError("Controller cell claim path was unlinked while held.") from error
    _require(
        binding.get("semantics") == CELL_CLAIM_SEMANTICS
        and binding.get("device") == current.st_dev
        and binding.get("inode") == current.st_ino
        and stat.S_ISREG(current.st_mode)
        and current.st_uid == os.getuid()
        and current.st_nlink == 1
        and stat.S_IMODE(current.st_mode) == 0o600,
        "Controller cell claim path was replaced or became unsafe.",
    )


@contextmanager
def _exclusive_cell_claim(
    output_dir: Path,
    *,
    coordinate: ShardCoordinate,
    launch_nonce: str,
    worker_index: int | None = None,
    worker_count: int | None = None,
) -> Iterator[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_path = output_dir / CELL_CLAIM_NAME
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Cell claims require O_NOFOLLOW.")
    flags = (
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow)
    )
    try:
        descriptor = os.open(claim_path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(
            f"Controller cell already has an orphan or active claim: {coordinate.key}"
        ) from error
    payload = {
        "schema_version": 1,
        "semantics": CELL_CLAIM_SEMANTICS,
        "coordinate": coordinate.payload,
        "launch_nonce": launch_nonce,
        "pid": os.getpid(),
        "process_start_ticks": _process_start_ticks(os.getpid()),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "created_time_ns": time.time_ns(),
    }
    metadata = os.fstat(descriptor)
    binding = {
        "path": str(claim_path),
        "semantics": CELL_CLAIM_SEMANTICS,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
    }
    release_claim = False
    try:
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and stat.S_IMODE(metadata.st_mode) == 0o600
            and metadata.st_nlink == 1,
            "Controller cell claim metadata is unsafe.",
        )
        _write_fd_json(descriptor, payload)
        _assert_cell_claim_binding(binding)
        yield binding
        release_claim = True
    finally:
        try:
            _assert_cell_claim_binding(binding)
            held = os.fstat(descriptor)
            _require(
                (held.st_dev, held.st_ino) == (metadata.st_dev, metadata.st_ino)
                and held.st_nlink == 1,
                "Controller cell claim descriptor drifted while held.",
            )
            if release_claim:
                claim_path.unlink()
                _require(
                    os.fstat(descriptor).st_nlink == 0 and not os.path.lexists(claim_path),
                    "Controller cell claim release did not remove the held path.",
                )
        finally:
            os.close(descriptor)


def _sha256_fd(file_descriptor: int) -> str:
    return attestation.checksum_fd(file_descriptor)


def _file_binding(path: Path, *, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        binding: dict[str, Any] = {
            "path": str(opened.path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
        }
        if payload is not None:
            binding["payload_sha256"] = payload.get("payload_sha256")
            envelope = payload.get("attestation")
            _require(isinstance(envelope, Mapping), f"Attestation missing from {path}.")
            binding["attestation_mac"] = cast(Mapping[str, Any], envelope).get("mac")
        opened.assert_unchanged()
        return binding
    finally:
        opened.close()


def _worker_ledger_root_binding(output_root: Path, *, worker_count: int) -> dict[str, Any] | None:
    if worker_count == 1:
        return None
    _validate_worker_identity(worker_index=0, worker_count=worker_count)
    root = _exact_resolved_path(
        _worker_ledger_root(output_root), label="Controller worker-ledger root"
    )
    return {
        "path": str(root),
        "semantics": WORKER_LEDGER_ROOT_SEMANTICS,
        "path_semantics": WORKER_LEDGER_PATH_SEMANTICS,
        "worker_count": worker_count,
    }


def _worker_ledger_registry(
    output_root: Path,
    *,
    worker_count: int,
    completed_by_worker: Mapping[int, int],
) -> list[dict[str, Any]]:
    root = _worker_ledger_root(output_root)
    if worker_count == 1:
        _require(
            not completed_by_worker and not root.exists(),
            "Single-worker matrix requires an exact empty worker-ledger registry.",
        )
        return []
    _validate_worker_identity(worker_index=0, worker_count=worker_count)
    _require(
        all(
            type(index) is int and 0 <= index < worker_count and type(count) is int and count >= 0
            for index, count in completed_by_worker.items()
        ),
        "Worker-ledger registry input is invalid.",
    )
    expected = {
        _worker_ledger_path(output_root, worker_index=index, worker_count=worker_count): index
        for index in completed_by_worker
    }
    if not expected:
        if root.exists():
            _require(
                root.is_dir() and not root.is_symlink() and not any(root.iterdir()),
                "Empty worker-ledger registry has a non-empty or unsafe root.",
            )
        return []
    _require(root.is_dir() and not root.is_symlink(), "Worker-ledger root is missing or unsafe.")
    observed = set(root.iterdir())
    _require(
        all(not item.is_symlink() and item.is_file() for item in observed)
        and {_absolute(item) for item in observed} == set(expected),
        "Worker-ledger root inventory differs from the public registry.",
    )
    root_binding = _worker_ledger_root_binding(output_root, worker_count=worker_count)
    _require(root_binding is not None, "Distributed worker-ledger root binding is missing.")
    result: list[dict[str, Any]] = []
    for path, index in sorted(expected.items(), key=lambda item: item[1]):
        payload = _load_json_nofollow(path, label="controller worker ledger registry member")
        envelope = payload.get("attestation")
        _require(isinstance(envelope, Mapping), "Worker-ledger attestation identity is missing.")
        completed = completed_by_worker[index]
        assigned_count = len(_assigned_coordinates(worker_index=index, worker_count=worker_count))
        expected_status = "terminal" if completed == assigned_count else "in_progress"
        _require(
            payload.get("worker_index") == index
            and payload.get("worker_count") == worker_count
            and payload.get("assignment_rule") == WORKER_ASSIGNMENT_RULE
            and payload.get("completed_shards") == completed
            and payload.get("assigned_coordinate_count") == assigned_count
            and payload.get("status") == expected_status,
            "Worker-ledger registry member identity or progress drifted.",
        )
        file_binding = _file_binding(path, payload=payload)
        result.append(
            {
                "worker_index": index,
                "worker_count": worker_count,
                "assignment_rule": WORKER_ASSIGNMENT_RULE,
                "root": dict(cast(Mapping[str, Any], root_binding)),
                "path_semantics": WORKER_LEDGER_PATH_SEMANTICS,
                "relative_path": path.name,
                **file_binding,
                "payload_sha256": payload.get("payload_sha256"),
                "attestation": dict(cast(Mapping[str, Any], envelope)),
                "status": expected_status,
                "completed_shards": completed,
                "assigned_coordinate_count": assigned_count,
            }
        )
    return result


def _load_json_nofollow(path: Path, *, label: str) -> dict[str, Any]:
    opened = attestation.open_regular_nofollow(path)
    try:
        try:
            payload = json.loads(opened.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid {label}: {path}") from error
        _require(isinstance(payload, dict), f"{label} must be a JSON object: {path}")
        opened.assert_unchanged()
        return payload
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
        result, trust_root=trust_root, purpose=MATRIX_ATTESTATION_PURPOSE
    )
    return result


def _verify_attested_payload(
    payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> None:
    digest = payload.get("payload_sha256")
    source = dict(payload)
    source.pop("attestation", None)
    source.pop("payload_sha256", None)
    _require(
        contract.is_sha256(digest) and digest == contract.json_digest(source),
        "Controller matrix payload digest drifted.",
    )
    envelope = payload.get("attestation")
    _require(isinstance(envelope, Mapping), "Controller matrix attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=MATRIX_ATTESTATION_PURPOSE,
    )


def _attested_worker_payload(
    payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> dict[str, Any]:
    result = _digest_bound_payload(payload)
    result["attestation"] = attestation.attest_payload(
        result,
        trust_root=trust_root,
        purpose=WORKER_ATTESTATION_PURPOSE,
    )
    return result


def _verify_worker_payload(
    payload: Mapping[str, Any], *, trust_root: attestation.TrustRoot
) -> None:
    digest = payload.get("payload_sha256")
    source = dict(payload)
    envelope = source.pop("attestation", None)
    source.pop("payload_sha256", None)
    _require(
        contract.is_sha256(digest) and digest == contract.json_digest(source),
        "Controller worker-ledger payload digest drifted.",
    )
    _require(isinstance(envelope, Mapping), "Controller worker-ledger attestation is missing.")
    semantic = dict(payload)
    semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        cast(Mapping[str, Any], envelope),
        trust_root=trust_root,
        purpose=WORKER_ATTESTATION_PURPOSE,
    )


def _canonical_evaluator(candidate: Path) -> Path:
    _require(not candidate.is_symlink(), "Evaluator script may not be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
        canonical = EVALUATOR_SCRIPT.resolve(strict=True)
    except OSError as error:
        raise ValueError("Canonical direct evaluator is missing or inaccessible.") from error
    _require(resolved == canonical, "Evaluator must be the manifest-bound canonical script.")
    return canonical


def _open_evaluator(
    canonical: Path, *, expected: CanonicalEvaluatorSnapshot | None = None
) -> tuple[int, CanonicalEvaluatorSnapshot]:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Stable evaluator execution requires O_NOFOLLOW.")
    descriptor = os.open(
        canonical,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
    )
    try:
        opened = os.fstat(descriptor)
        current = os.stat(canonical, follow_symlinks=False)
        _require(stat.S_ISREG(opened.st_mode), "Canonical evaluator is not regular.")
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino),
            "Canonical evaluator changed while opening.",
        )
        snapshot = CanonicalEvaluatorSnapshot(
            path=str(canonical),
            sha256=_sha256_fd(descriptor),
            bytes=opened.st_size,
            device=opened.st_dev,
            inode=opened.st_ino,
            mode=opened.st_mode,
            mtime_ns=opened.st_mtime_ns,
            ctime_ns=opened.st_ctime_ns,
        )
        if expected is not None:
            _require(snapshot == expected, "Canonical evaluator changed after preflight.")
        return descriptor, snapshot
    except BaseException:
        os.close(descriptor)
        raise


def _sealed_evaluator_copy(descriptor: int, *, expected: CanonicalEvaluatorSnapshot) -> int:
    create = getattr(os, "memfd_create", None)
    allow_sealing = getattr(os, "MFD_ALLOW_SEALING", None)
    _require(
        create is not None and allow_sealing is not None,
        "Stable evaluator execution requires sealed memfd support.",
    )
    sealed = cast(Callable[[str, int], int], create)(
        "adaptive-v4-canonical-direct-evaluator",
        cast(int, getattr(os, "MFD_CLOEXEC", 0)) | cast(int, allow_sealing),
    )
    try:
        offset = 0
        while offset < expected.bytes:
            chunk = os.pread(descriptor, min(1024 * 1024, expected.bytes - offset), offset)
            _require(bool(chunk), "Canonical evaluator truncated during snapshotting.")
            written = 0
            while written < len(chunk):
                count = os.write(sealed, chunk[written:])
                _require(count > 0, "Sealed evaluator write stalled.")
                written += count
            offset += len(chunk)
        _require(_sha256_fd(sealed) == expected.sha256, "Sealed evaluator digest drifted.")
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(sealed, fcntl.F_ADD_SEALS, seals)
        _require(
            fcntl.fcntl(sealed, fcntl.F_GET_SEALS) & seals == seals,
            "Evaluator memfd was not sealed.",
        )
        os.lseek(sealed, 0, os.SEEK_SET)
        return sealed
    except BaseException:
        os.close(sealed)
        raise


def _assert_open_evaluator(descriptor: int, *, expected: CanonicalEvaluatorSnapshot) -> None:
    opened = os.fstat(descriptor)
    current = os.stat(expected.path, follow_symlinks=False)
    _require(
        (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mode,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            _sha256_fd(descriptor),
        )
        == (
            expected.device,
            expected.inode,
            expected.bytes,
            expected.mode,
            expected.mtime_ns,
            expected.ctime_ns,
            expected.sha256,
        ),
        "Open canonical evaluator changed during execution.",
    )
    _require(
        (current.st_dev, current.st_ino) == (expected.device, expected.inode),
        "Canonical evaluator path changed during execution.",
    )


def _run_evaluator_from_snapshot(
    command: Sequence[str],
    *,
    canonical: Path,
    expected: CanonicalEvaluatorSnapshot,
    trust_root: attestation.TrustRoot,
    gpu_lease: GPULockLease,
    device_guard_lease: GPULockLease,
    projected_shards: int,
    projected_token_rows: int,
) -> subprocess.CompletedProcess[Any]:
    _require(
        type(projected_shards) is int and 1 <= projected_shards <= EXPECTED_SHARDS,
        "Projected remaining shards are outside the frozen matrix.",
    )
    _require(
        type(projected_token_rows) is int
        and 1 <= projected_token_rows <= contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES,
        "Projected remaining decode-token weight is outside the frozen matrix.",
    )
    gpu_lease.assert_held()
    device_guard_lease.assert_held()
    gpu_descriptor = gpu_lease.fileno()
    device_guard_descriptor = device_guard_lease.fileno()
    inherited_gpu_descriptors = tuple(sorted({gpu_descriptor, device_guard_descriptor}))
    _require(
        all(descriptor >= 0 for descriptor in inherited_gpu_descriptors),
        "Evaluator launch lacks valid inherited GPU lease FDs.",
    )
    descriptor, snapshot = _open_evaluator(canonical, expected=expected)
    sealed: int | None = None
    key_descriptor: int | None = None
    try:
        sealed = _sealed_evaluator_copy(descriptor, expected=snapshot)
        key_descriptor = attestation.create_sealed_key_fd(trust_root)
        environment = os.environ.copy()
        environment.pop(attestation.KEY_PATH_ENV, None)
        environment[EVALUATOR_FD_ENV] = str(sealed)
        environment[attestation.KEY_FD_ENV] = str(key_descriptor)
        environment[STORAGE_PROJECTED_SHARDS_ENV] = str(projected_shards)
        environment[STORAGE_PROJECTED_TOKEN_ROWS_ENV] = str(projected_token_rows)
        result = subprocess.run(
            list(command),
            check=False,
            pass_fds=(sealed, key_descriptor, *inherited_gpu_descriptors),
            env=environment,
        )
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
        _assert_open_evaluator(descriptor, expected=snapshot)
        return result
    finally:
        if key_descriptor is not None:
            os.close(key_descriptor)
        if sealed is not None:
            os.close(sealed)
        os.close(descriptor)


def _top_p_artifact_path(
    root: Path, *, scale: str, training_seed: int, budget: str, comparator: str
) -> Path:
    try:
        module = importlib.import_module("validate_p2_direct_top_p_physical_match")
    except ModuleNotFoundError as error:
        if error.name != "validate_p2_direct_top_p_physical_match":
            raise
    else:
        helper = getattr(module, "artifact_path", None)
        if callable(helper):
            return cast(Path, helper(root, scale, training_seed, budget, comparator))
    _require(comparator in contract.SENSITIVITY_COMPARATOR_ARMS, "Unknown top-p comparator.")
    return root / scale / f"seed-{training_seed}" / budget / f"{comparator}.json"


def _artifact_input_binding(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    binding = _file_binding(path, payload=payload)
    binding["experiment_id"] = payload.get("experiment_id")
    return binding


def _load_terminal_top_p_matrix(
    *,
    manifest_path: Path,
    training_output_root: Path,
    calibration_output_root: Path,
    top_p_output_root: Path,
    attestation_key_path: Path | None,
    expected_context: training_matrix.FrozenContext,
    expected_trust_root: attestation.TrustRoot,
    gpu_lease: GPULockLease | None = None,
    device_guard_lease: GPULockLease | None = None,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    _require(
        (gpu_lease is None) == (device_guard_lease is None),
        "Top-p controller validation requires both scheduler and physical-device leases.",
    )
    if gpu_lease is not None and device_guard_lease is not None:
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
    prerequisites = top_p_matrix.load_and_validate_prerequisites(
        manifest_path=manifest_path,
        training_output_root=training_output_root,
        calibration_output_root=calibration_output_root,
        output_root=top_p_output_root,
        attestation_key_path=attestation_key_path,
    )
    _require(
        prerequisites.context == expected_context
        and prerequisites.trust_root.key_id == expected_trust_root.key_id,
        "Top-p matrix trust root or frozen context differs from the quality matrix.",
    )
    top_p_matrix._assert_implementation_binding(expected_context)
    canonical_generator = top_p_matrix._canonical_generator(top_p_matrix.GENERATOR_SCRIPT)
    opened, generator_snapshot = top_p_matrix._open_generator(canonical_generator)
    opened.close()
    summary_path = top_p_output_root / top_p_matrix.MATRIX_SUMMARY_NAME
    _require(summary_path.is_file(), "Terminal top-p physical matrix ledger is missing.")
    payload = _load_json_nofollow(summary_path, label="top-p physical matrix ledger")
    top_p_lock_binding = top_p_matrix._matrix_lock_binding(
        top_p_matrix._matrix_lock_path(top_p_output_root)
    )
    if gpu_lease is None:
        records = top_p_matrix.validate_matrix_summary_archived(
            payload,
            output_root=top_p_output_root,
            prerequisites=prerequisites,
            generator_script=canonical_generator,
            generator_binding=generator_snapshot.public_binding,
            matrix_lock_binding=top_p_lock_binding,
            verify_artifacts=True,
        )
    else:
        records = top_p_matrix.validate_matrix_summary_controller_compatible(
            payload,
            output_root=top_p_output_root,
            prerequisites=prerequisites,
            generator_script=canonical_generator,
            generator_binding=generator_snapshot.public_binding,
            matrix_lock_binding=top_p_lock_binding,
            gpu_lease=gpu_lease,
            device_guard_lease=cast(GPULockLease, device_guard_lease),
            verify_artifacts=True,
        )
        gpu_lease.assert_held()
        cast(GPULockLease, device_guard_lease).assert_held()
    # Closed-world verification is intentionally repeated here even when the public validator
    # performs it, so a downstream quality launch cannot accept hand-picked valid artifacts.
    top_p_matrix._preflight_output_tree(
        output_root=top_p_output_root,
        matrix_summary=summary_path,
        completed_cells=len(records),
    )
    verification, _ = top_p_matrix._open_generator(canonical_generator, expected=generator_snapshot)
    verification.close()
    _require(
        payload.get("experiment_id") == top_p_matrix.EXPERIMENT_ID
        and payload.get("artifact_type") == top_p_matrix.ARTIFACT_TYPE
        and isinstance(payload.get("attestation"), Mapping)
        and payload.get("status") == "terminal"
        and payload.get("completed_cells") == top_p_matrix.EXPECTED_CELLS
        and len(records) == top_p_matrix.EXPECTED_CELLS == 40,
        "Quality launch requires the terminal exact 40-cell top-p matrix ledger.",
    )
    training_matrix.assert_environment_unchanged(expected_context)
    return summary_path, payload, records


def load_and_validate_prerequisites(
    *,
    manifest_path: Path,
    training_output_root: Path,
    calibration_output_root: Path,
    top_p_output_root: Path,
    output_root: Path,
    attestation_key_path: Path | None,
    gpu_lease: GPULockLease | None = None,
    device_guard_lease: GPULockLease | None = None,
) -> FrozenPrerequisites:
    """Authenticate every training/calibration/match input before the first quality launch."""

    _require(
        (gpu_lease is None) == (device_guard_lease is None),
        "Controller prerequisites require both scheduler and physical-device leases.",
    )
    if gpu_lease is not None and device_guard_lease is not None:
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
    context = training_matrix.establish_frozen_context(manifest_path)
    raw_attestation = context.manifest_binding.get("attestation")
    _require(isinstance(raw_attestation, Mapping), "Manifest attestation binding is missing.")
    expected_key_id = cast(Mapping[str, Any], raw_attestation).get("key_id")
    _require(contract.is_sha256(expected_key_id), "Manifest attestation key ID is invalid.")
    roots = (training_output_root, calibration_output_root, top_p_output_root, output_root)
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
    _require(
        calibration_ledger.get("status") == "terminal"
        and calibration_ledger.get("terminal_decision") == "GO"
        and len(calibration_records) == len(FROZEN_SCALES) * len(FROZEN_TRAINING_SEEDS),
        "Quality launch requires a terminal all-GO calibration matrix.",
    )

    raw_training_environment = training_ledger.get("execution_environment")
    _require(
        isinstance(raw_training_environment, Mapping),
        "Terminal training matrix execution environment is missing.",
    )
    frozen_environment_projection = (
        execution_environment.controller_compatible_environment_projection(
            cast(Mapping[str, Any], raw_training_environment)
        )
    )

    top_p_ledger_path, top_p_ledger, top_p_records = _load_terminal_top_p_matrix(
        manifest_path=manifest_path,
        training_output_root=training_output_root,
        calibration_output_root=calibration_output_root,
        top_p_output_root=top_p_output_root,
        attestation_key_path=attestation_key_path,
        expected_context=context,
        expected_trust_root=trust_root,
        gpu_lease=gpu_lease,
        device_guard_lease=device_guard_lease,
    )

    bundles: dict[tuple[str, int, str], InputBundle] = {}
    for scale in FROZEN_SCALES:
        for training_seed in FROZEN_TRAINING_SEEDS:
            calibration_path = calibration_matrix._artifact_path(
                calibration_output_root, scale, training_seed
            )
            calibration_payload = _load_json_nofollow(
                calibration_path, label="direct calibration artifact"
            )
            calibration.validate_calibration_artifact(
                calibration_payload, verify_bindings=True, trust_root=trust_root
            )
            _require(
                calibration_payload.get("terminal_decision") == "GO",
                f"Direct calibration is not GO: {scale}/{training_seed}.",
            )
            checkpoint = calibration_payload.get("checkpoint")
            training_summary = calibration_payload.get("training_summary")
            _require(isinstance(checkpoint, Mapping), "Calibration checkpoint binding is missing.")
            _require(
                isinstance(training_summary, Mapping),
                "Calibration training-summary binding is missing.",
            )
            checkpoint_map = cast(Mapping[str, Any], checkpoint)
            training_summary_map = cast(Mapping[str, Any], training_summary)
            training_matrix_binding = training_summary_map.get("terminal_matrix_ledger")
            _require(
                isinstance(training_matrix_binding, Mapping),
                "Calibration terminal-training-ledger binding is missing.",
            )
            for budget in FROZEN_BUDGETS:
                top_p_paths = {
                    comparator: _top_p_artifact_path(
                        top_p_output_root,
                        scale=scale,
                        training_seed=training_seed,
                        budget=budget,
                        comparator=comparator,
                    )
                    for comparator in contract.SENSITIVITY_COMPARATOR_ARMS
                }
                matches = {
                    comparator: _load_json_nofollow(path, label="top-p match artifact")
                    for comparator, path in top_p_paths.items()
                }
                arms, _metadata = contract.build_direct_controller_arms(
                    calibration_payload,
                    budget,
                    trust_root=trust_root,
                    expected_scale=scale,
                    expected_training_seed=training_seed,
                    expected_global_block_budget=contract.DIRECT_GLOBAL_BLOCK_BUDGETS[scale][
                        budget
                    ],
                    expected_csa_layers=contract.DIRECT_CSA_LAYERS_BY_SCALE[scale],
                    comparator_matches=matches,
                )
                _require(tuple(arms) == ARM_NAMES, "Validated quality arm inventory drifted.")
                input_source = {
                    "source": calibration_payload["source"],
                    "manifest": calibration_payload["manifest"],
                    "checkpoint": dict(checkpoint_map),
                    "training_summary": dict(training_summary_map),
                    "calibration_artifact": _artifact_input_binding(
                        calibration_path, calibration_payload
                    ),
                    "top_p_match_artifacts": {
                        comparator: _artifact_input_binding(
                            top_p_paths[comparator], matches[comparator]
                        )
                        for comparator in contract.SENSITIVITY_COMPARATOR_ARMS
                    },
                }
                bundle_binding = {
                    **input_source,
                    "input_binding_digest": contract.json_digest(input_source),
                }
                bundles[(scale, training_seed, budget)] = InputBundle(
                    coordinate=(scale, training_seed, budget),
                    checkpoint_path=Path(cast(str, checkpoint_map["path"])),
                    training_summary_path=Path(cast(str, training_summary_map["path"])),
                    training_matrix_summary_path=Path(
                        cast(str, cast(Mapping[str, Any], training_matrix_binding)["path"])
                    ),
                    calibration_path=calibration_path,
                    top_p_05_path=top_p_paths["fixed-top-p-0.5+pins"],
                    top_p_08_path=top_p_paths["fixed-top-p-0.8+pins"],
                    calibration_payload=calibration_payload,
                    binding=bundle_binding,
                )
    _require(len(bundles) == 20, "Direct prerequisite bundle inventory is incomplete.")
    training_matrix.assert_environment_unchanged(context)
    if gpu_lease is not None and device_guard_lease is not None:
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
    return FrozenPrerequisites(
        context=context,
        trust_root=trust_root,
        bundles=bundles,
        public_binding={
            "manifest": context.manifest_binding,
            "training_matrix": _artifact_input_binding(training_ledger_path, training_ledger),
            "calibration_matrix": _artifact_input_binding(
                calibration_ledger_path, calibration_ledger
            ),
            "top_p_matrix": _artifact_input_binding(top_p_ledger_path, top_p_ledger),
            "top_p_root": str(top_p_output_root),
            "execution_environment_projection": frozen_environment_projection,
            "validated_training_cells": len(ledger_records),
            "validated_calibration_cells": len(calibration_records),
            "validated_top_p_cells": len(top_p_records),
            "validated_scale_seed_budget_bundles": len(bundles),
        },
    )


def build_evaluator_command(
    *,
    evaluator_script: Path,
    envelope_path: Path,
    manifest_path: Path,
    coordinate: ShardCoordinate,
    inputs: InputBundle,
    launch_nonce: str,
    gpu_lease_binding: Mapping[str, Any],
) -> list[str]:
    canonical = _canonical_evaluator(evaluator_script)
    validated_gpu_binding = _validate_gpu_lease_binding(gpu_lease_binding)
    routing_identity = cast(
        Mapping[str, Any], validated_gpu_binding["selected_device_routing_identity"]
    )
    command = [
        sys.executable,
        "-I",
        "-c",
        EVALUATOR_FD_BOOTSTRAP,
        str(canonical),
        "--checkpoint",
        str(inputs.checkpoint_path.resolve()),
        "--training-summary",
        str(inputs.training_summary_path.resolve()),
        "--training-matrix-summary",
        str(inputs.training_matrix_summary_path.resolve()),
        "--calibration",
        str(inputs.calibration_path.resolve()),
        "--top-p-match-05",
        str(inputs.top_p_05_path.resolve()),
        "--top-p-match-08",
        str(inputs.top_p_08_path.resolve()),
        "--manifest",
        str(manifest_path.resolve()),
        "--scale",
        coordinate.scale,
        "--training-seed",
        str(coordinate.training_seed),
        "--budget",
        coordinate.budget,
        "--family",
        coordinate.family,
        "--context",
        str(coordinate.context),
        "--replicate",
        str(coordinate.replicate),
        "--output",
        str(envelope_path),
        "--device",
        cast(str, validated_gpu_binding["device_argument"]),
        "--expected-device-identity-type",
        cast(str, routing_identity["identity_type"]),
        "--expected-device-identity",
        cast(str, routing_identity["identity"]),
        "--dtype",
        DTYPE,
    ]
    # The nonce option is part of the finalized evaluator interface. Keeping it last makes the
    # adapter easy to compare in audit records and impossible to confuse with a coordinate value.
    command.extend(("--launch-nonce", launch_nonce))
    return command


def _evaluator_module() -> Any:
    return importlib.import_module("evaluate_p2_direct_controller_shard")


def _envelope_sidecars(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    sidecars = payload.get("sidecars")
    if not isinstance(sidecars, Mapping):
        sidecars = payload.get("bundle_members")
    if not isinstance(sidecars, Mapping):
        raise ValueError("Evaluator sidecar inventory is missing.")
    _require(set(sidecars) == set(BUNDLE_KINDS), "Evaluator sidecar inventory drifted.")
    return sidecars


def _sidecar_relative_path(binding: Mapping[str, Any]) -> str:
    value = binding.get("relative_path")
    if not isinstance(value, str) or value != Path(value).name:
        raise ValueError("Unsafe sidecar path binding.")
    return value


def _validate_bundle_paths(
    envelope_path: Path,
    payload: Mapping[str, Any],
    *,
    coordinate: ShardCoordinate,
) -> dict[str, dict[str, Any]]:
    module = _evaluator_module()
    helper = getattr(module, "canonical_bundle_paths", None)
    _require(callable(helper), "Evaluator canonical-bundle helper is unavailable.")
    expected_paths = cast(Callable[[Path], Any], helper)(envelope_path)
    _require(
        isinstance(expected_paths, Mapping) and set(expected_paths) == {"envelope", *BUNDLE_KINDS},
        "Evaluator canonical-bundle path inventory drifted.",
    )
    result: dict[str, dict[str, Any]] = {}
    for kind, raw in _envelope_sidecars(payload).items():
        _require(isinstance(raw, Mapping), f"Evaluator {kind} sidecar binding is invalid.")
        binding = dict(cast(Mapping[str, Any], raw))
        relative = _sidecar_relative_path(binding)
        path = cast(Path, expected_paths[kind])
        _require(relative == path.name, f"Evaluator {kind} sidecar name drifted.")
        observed = _file_binding(path)
        _require(
            binding.get("compressed_sha256") == observed["sha256"]
            and binding.get("compressed_bytes") == observed["bytes"],
            f"Evaluator {kind} sidecar bytes drifted.",
        )
        result[kind] = {**binding, "path": str(path)}
    return result


def load_and_validate_shard_bundle(
    envelope_path: Path,
    *,
    coordinate: ShardCoordinate,
    inputs: InputBundle,
    launch_nonce: str,
    trust_root: attestation.TrustRoot,
    gpu_lease_binding: Mapping[str, Any],
    execution_environment_projection: Mapping[str, Any],
) -> dict[str, Any]:
    module = _evaluator_module()
    loader = getattr(module, "load_validated_direct_controller_shard", None)
    _require(callable(loader), "Direct evaluator safe shard loader is unavailable.")
    payload = cast(Callable[..., Any], loader)(
        envelope_path, verify_bindings=True, trust_root=trust_root
    )
    _require(isinstance(payload, dict), "Direct evaluator loader returned no envelope.")
    expected_gpu = _validate_gpu_lease_binding(gpu_lease_binding)
    environment = payload.get("environment")
    selected_class = cast(Mapping[str, Any], expected_gpu["selected_device_class"])
    _require(
        isinstance(environment, Mapping)
        and environment.get("device_argument") == expected_gpu["device_argument"]
        and environment.get("device_index") == expected_gpu["selected_device_logical_index"]
        and environment.get("selected_device_routing_identity")
        == expected_gpu["selected_device_routing_identity"]
        and environment.get("cuda_device_name") == selected_class["name"]
        and environment.get("cuda_capability") == selected_class["compute_capability"]
        and environment.get("cuda_total_memory_bytes") == selected_class["total_memory_bytes"]
        and environment.get("python") == execution_environment_projection.get("python_version")
        and environment.get("torch") == execution_environment_projection.get("torch_version")
        and environment.get("torch_cuda_version")
        == execution_environment_projection.get("cuda_runtime_version"),
        "Evaluator envelope runtime environment differs from its guarded worker device.",
    )
    _require(
        payload.get("coordinate") == coordinate.evaluator_payload,
        "Evaluator shard coordinate drifted.",
    )
    _require(payload.get("launch_nonce") == launch_nonce, "Evaluator launch nonce drifted.")
    _require(
        payload.get("terminal_decision") in {"INTEGRITY-PASS", "INTEGRITY-FAIL"},
        "Evaluator terminal integrity decision is invalid.",
    )
    workload = payload.get("workload_contract")
    arm_contract = payload.get("arm_contract")
    _require(isinstance(workload, Mapping), "Evaluator workload contract is missing.")
    _require(isinstance(arm_contract, Mapping), "Evaluator arm contract is missing.")
    _require(
        cast(Mapping[str, Any], workload).get("examples") == EXAMPLES_PER_SHARD,
        "Shard size drifted.",
    )
    _require(
        tuple(cast(Mapping[str, Any], arm_contract).get("all_arms", ())) == ARM_NAMES,
        "Evaluator arm inventory drifted.",
    )
    _require(payload.get("inputs") == dict(inputs.binding), "Evaluator input binding drifted.")
    sidecars = _validate_bundle_paths(envelope_path, payload, coordinate=coordinate)
    return {**payload, "_validated_sidecars": sidecars}


def _expected_exit_code(decision: str) -> int:
    _require(
        decision in {"INTEGRITY-PASS", "INTEGRITY-FAIL"},
        "Unknown evaluator integrity decision.",
    )
    return 0 if decision == "INTEGRITY-PASS" else 2


def _run_record(
    payload: Mapping[str, Any],
    *,
    coordinate: ShardCoordinate,
    envelope_path: Path,
    inputs: InputBundle,
    launch_nonce: str,
    command: Sequence[str],
) -> dict[str, Any]:
    sidecars = payload.get("_validated_sidecars")
    _require(isinstance(sidecars, Mapping), "Validated evaluator sidecars are missing.")
    envelope_binding = _file_binding(envelope_path, payload=payload)
    decision = cast(str, payload["terminal_decision"])
    storage = payload.get("storage", payload.get("storage_metrics", {}))
    _require(isinstance(storage, Mapping), "Evaluator storage metrics are invalid.")
    return {
        **coordinate.payload,
        "coordinate_key": coordinate.key,
        "launch_nonce": launch_nonce,
        "status": "terminal",
        "integrity_decision": decision,
        "evaluator_exit_code": _expected_exit_code(decision),
        "command": list(command),
        "inputs": dict(inputs.binding),
        "artifact_bundle": {
            "envelope": envelope_binding,
            "sidecars": dict(cast(Mapping[str, Any], sidecars)),
            "storage": dict(cast(Mapping[str, Any], storage)),
        },
    }


def _storage_observations(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for record in records:
        artifact = record.get("artifact_bundle")
        _require(isinstance(artifact, Mapping), "Matrix artifact bundle binding is invalid.")
        artifact_map = cast(Mapping[str, Any], artifact)
        envelope = artifact_map.get("envelope")
        sidecars = artifact_map.get("sidecars")
        storage = artifact_map.get("storage")
        _require(isinstance(envelope, Mapping), "Matrix envelope binding is invalid.")
        _require(isinstance(sidecars, Mapping), "Matrix sidecar binding is invalid.")
        _require(isinstance(storage, Mapping), "Matrix evaluator storage evidence is invalid.")
        sidecars_map = cast(Mapping[str, Any], sidecars)
        storage_map = cast(Mapping[str, Any], storage)
        _require(set(sidecars_map) == set(BUNDLE_KINDS), "Matrix sidecar inventory drifted.")

        def sidecar_int(kind: str, field: str, inventory: Mapping[str, Any] = sidecars_map) -> int:
            binding = inventory.get(kind)
            _require(isinstance(binding, Mapping), f"Matrix {kind} sidecar binding is invalid.")
            value = cast(Mapping[str, Any], binding).get(field)
            _require(
                type(value) is int and value >= 0,
                f"Matrix {kind} sidecar {field} is invalid.",
            )
            return cast(int, value)

        sidecars_compressed = sum(sidecar_int(kind, "compressed_bytes") for kind in BUNDLE_KINDS)
        sidecars_uncompressed = sum(
            sidecar_int(kind, "uncompressed_bytes") for kind in BUNDLE_KINDS
        )
        token_compressed = sidecar_int("tokens", "compressed_bytes")
        token_uncompressed = sidecar_int("tokens", "uncompressed_bytes")
        token_binding = cast(Mapping[str, Any], sidecars_map["tokens"])
        observed_rows = token_binding.get("row_count")
        envelope_bytes = cast(Mapping[str, Any], envelope).get("bytes")
        _require(
            type(observed_rows) is int and observed_rows >= 0,
            "Matrix token sidecar row count is invalid.",
        )
        _require(
            type(envelope_bytes) is int and envelope_bytes >= 1,
            "Matrix envelope byte count is invalid.",
        )
        _require(
            storage_map.get("actual_sidecars_compressed_bytes") == sidecars_compressed
            and storage_map.get("actual_sidecars_uncompressed_bytes") == sidecars_uncompressed
            and storage_map.get("actual_token_sidecar_compressed_bytes") == token_compressed
            and storage_map.get("actual_token_sidecar_uncompressed_bytes") == token_uncompressed
            and storage_map.get("observed_token_rows") == observed_rows,
            "Matrix evaluator storage evidence drifted from its sidecars.",
        )
        if observed_rows:
            _require(
                storage_map.get("token_rate_compressed_numerator_bytes") == token_compressed
                and storage_map.get("token_rate_uncompressed_numerator_bytes") == token_uncompressed
                and storage_map.get("token_rate_denominator_rows") == observed_rows,
                "Matrix evaluator token-rate evidence drifted.",
            )
            rate_compressed_numerator: int | None = token_compressed
            rate_uncompressed_numerator: int | None = token_uncompressed
            rate_denominator: int | None = cast(int, observed_rows)
        else:
            _require(
                storage_map.get("token_rate_compressed_numerator_bytes") is None
                and storage_map.get("token_rate_uncompressed_numerator_bytes") is None
                and storage_map.get("token_rate_denominator_rows") is None,
                "Zero-token evaluator evidence fabricated a token rate.",
            )
            rate_compressed_numerator = None
            rate_uncompressed_numerator = None
            rate_denominator = None
        observations.append(
            {
                "coordinate_key": record.get("coordinate_key"),
                "envelope_bytes": cast(int, envelope_bytes),
                "sidecars_compressed_bytes": sidecars_compressed,
                "sidecars_uncompressed_bytes": sidecars_uncompressed,
                "fixed_sidecars_compressed_bytes": sidecars_compressed - token_compressed,
                "fixed_sidecars_uncompressed_bytes": sidecars_uncompressed - token_uncompressed,
                "token_rate_compressed_numerator_bytes": rate_compressed_numerator,
                "token_rate_uncompressed_numerator_bytes": rate_uncompressed_numerator,
                "token_rate_denominator_rows": rate_denominator,
            }
        )
    return observations


def _maximum_rate(
    observations: Sequence[Mapping[str, Any]], *, numerator_field: str
) -> tuple[int, int] | None:
    result: tuple[int, int] | None = None
    for observation in observations:
        numerator = observation.get(numerator_field)
        denominator = observation.get("token_rate_denominator_rows")
        if numerator is None and denominator is None:
            continue
        _require(
            type(numerator) is int
            and numerator >= 0
            and type(denominator) is int
            and denominator > 0,
            "Matrix token-rate observation is invalid.",
        )
        candidate = (cast(int, numerator), cast(int, denominator))
        if result is None or candidate[0] * result[1] > result[0] * candidate[1]:
            result = candidate
    return result


def _ceil_rate(rate: tuple[int, int], rows: int) -> int:
    _require(type(rows) is int and rows >= 0, "Storage projection row count is invalid.")
    return (rate[0] * rows + rate[1] - 1) // rate[1]


def _storage_rollup(
    records: Sequence[Mapping[str, Any]],
    *,
    remaining_coordinates: Sequence[ShardCoordinate] | None = None,
) -> dict[str, Any]:
    observations = _storage_observations(records)
    remaining_items = (
        tuple(coordinates()[len(records) :])
        if remaining_coordinates is None
        else tuple(remaining_coordinates)
    )
    _require(
        len(observations) + len(remaining_items) == EXPECTED_SHARDS,
        "Storage observations and remaining coordinates do not cover the frozen matrix.",
    )
    remaining = len(remaining_items)
    remaining_token_rows = sum(expected_decode_token_rows(item) for item in remaining_items)
    actual_sidecars_compressed = sum(
        cast(int, item["sidecars_compressed_bytes"]) for item in observations
    )
    actual_sidecars_uncompressed = sum(
        cast(int, item["sidecars_uncompressed_bytes"]) for item in observations
    )
    actual_envelopes = sum(cast(int, item["envelope_bytes"]) for item in observations)
    maximum_fixed_compressed = (
        max(cast(int, item["fixed_sidecars_compressed_bytes"]) for item in observations)
        if observations
        else None
    )
    maximum_fixed_uncompressed = (
        max(cast(int, item["fixed_sidecars_uncompressed_bytes"]) for item in observations)
        if observations
        else None
    )
    maximum_envelope = (
        max(cast(int, item["envelope_bytes"]) for item in observations) if observations else None
    )
    compressed_rate = _maximum_rate(
        observations, numerator_field="token_rate_compressed_numerator_bytes"
    )
    uncompressed_rate = _maximum_rate(
        observations, numerator_field="token_rate_uncompressed_numerator_bytes"
    )
    if remaining == 0:
        estimated_compressed: int | None = 0
        estimated_uncompressed: int | None = 0
        estimate_method: str | None = "terminal-no-remaining-coordinates-v1"
    elif compressed_rate is not None and uncompressed_rate is not None:
        _require(
            maximum_fixed_compressed is not None
            and maximum_fixed_uncompressed is not None
            and maximum_envelope is not None,
            "Token-rate observations lack fixed/envelope observations.",
        )
        fixed_compressed = cast(int, maximum_fixed_compressed)
        fixed_uncompressed = cast(int, maximum_fixed_uncompressed)
        envelope_high_water = cast(int, maximum_envelope)
        estimated_compressed = (fixed_compressed + envelope_high_water) * remaining + _ceil_rate(
            compressed_rate, remaining_token_rows
        )
        estimated_uncompressed = (
            fixed_uncompressed + envelope_high_water
        ) * remaining + _ceil_rate(uncompressed_rate, remaining_token_rows)
        estimate_method = "component-wise-observed-high-water-rebased-to-remaining-grid-v1"
    else:
        estimated_compressed = None
        estimated_uncompressed = None
        estimate_method = None
    actual_bundle_compressed = actual_sidecars_compressed + actual_envelopes
    actual_bundle_uncompressed = actual_sidecars_uncompressed + actual_envelopes
    return {
        "bundle_count_expected": EXPECTED_SHARDS,
        "sidecar_count_expected": EXPECTED_SHARDS * len(BUNDLE_KINDS),
        "completed_bundle_count": len(observations),
        "completed_sidecar_count": len(observations) * len(BUNDLE_KINDS),
        "positive_token_rate_observation_count": sum(
            item["token_rate_denominator_rows"] is not None for item in observations
        ),
        "zero_token_rate_unavailable_observation_count": sum(
            item["token_rate_denominator_rows"] is None for item in observations
        ),
        "observation_components_digest": contract.json_digest(observations),
        "remaining_shards": remaining,
        "remaining_decode_token_rows": remaining_token_rows,
        "remaining_estimate_semantics": (
            "observed-component-high-water-planning-estimate;not-a-future-upper-bound;"
            "not-a-reservation;not-a-write-guarantee"
        ),
        "remaining_estimate_method": estimate_method,
        "maximum_observed_fixed_sidecars_compressed_bytes_per_shard": (maximum_fixed_compressed),
        "maximum_observed_fixed_sidecars_uncompressed_bytes_per_shard": (
            maximum_fixed_uncompressed
        ),
        "maximum_observed_envelope_bytes_per_shard": maximum_envelope,
        "maximum_observed_token_compressed_rate_numerator_bytes": (
            None if compressed_rate is None else compressed_rate[0]
        ),
        "maximum_observed_token_compressed_rate_denominator_rows": (
            None if compressed_rate is None else compressed_rate[1]
        ),
        "maximum_observed_token_uncompressed_rate_numerator_bytes": (
            None if uncompressed_rate is None else uncompressed_rate[0]
        ),
        "maximum_observed_token_uncompressed_rate_denominator_rows": (
            None if uncompressed_rate is None else uncompressed_rate[1]
        ),
        "observed_high_water_remaining_compressed_bytes": estimated_compressed,
        "observed_high_water_remaining_uncompressed_bytes": estimated_uncompressed,
        "completed_actual_sidecars_compressed_bytes": actual_sidecars_compressed,
        "completed_actual_sidecars_uncompressed_bytes": actual_sidecars_uncompressed,
        "completed_actual_envelopes_bytes": actual_envelopes,
        "completed_actual_bundle_compressed_bytes": actual_bundle_compressed,
        "completed_actual_bundle_uncompressed_bytes": actual_bundle_uncompressed,
        "ledger_actual_plus_observed_high_water_estimate_compressed_bytes": (
            None
            if estimated_compressed is None
            else actual_bundle_compressed + estimated_compressed
        ),
        "ledger_actual_plus_observed_high_water_estimate_uncompressed_bytes": (
            None
            if estimated_uncompressed is None
            else actual_bundle_uncompressed + estimated_uncompressed
        ),
        "next_launch_headroom_policy": {
            "method": NEXT_LAUNCH_HEADROOM_METHOD,
            "frozen_minimum_one_shard_headroom_bytes": (FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES),
            "outcome_independent_inputs": [
                "completed-record-byte-components",
                "next-coordinate-decode-row-count",
                "filesystem-available-bytes",
            ],
            "limitations": NEXT_LAUNCH_HEADROOM_LIMITATIONS,
        },
    }


def _headroom_signal_payload(
    *,
    records: Sequence[Mapping[str, Any]],
    coordinate: ShardCoordinate,
    filesystem_available_bytes: int,
) -> dict[str, Any]:
    _require(
        type(filesystem_available_bytes) is int and filesystem_available_bytes >= 0,
        "Filesystem available-byte observation is invalid.",
    )
    observations = _storage_observations(records)
    compressed_rate = _maximum_rate(
        observations, numerator_field="token_rate_compressed_numerator_bytes"
    )
    maximum_fixed = max(
        (cast(int, item["fixed_sidecars_compressed_bytes"]) for item in observations),
        default=0,
    )
    maximum_envelope = max((cast(int, item["envelope_bytes"]) for item in observations), default=0)
    next_rows = expected_decode_token_rows(coordinate)
    observed_estimate = (
        None
        if compressed_rate is None
        else maximum_fixed + maximum_envelope + _ceil_rate(compressed_rate, next_rows)
    )
    required = max(
        FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES,
        maximum_fixed + maximum_envelope,
        0 if observed_estimate is None else observed_estimate,
    )
    return {
        "schema_version": 1,
        "signal_type": "infrastructure-storage-headroom-pause-v1",
        "retryable": True,
        "coordinate_key": coordinate.key,
        "method": NEXT_LAUNCH_HEADROOM_METHOD,
        "outcome_independent": True,
        "completed_record_count": len(observations),
        "positive_token_rate_observation_count": sum(
            item["token_rate_denominator_rows"] is not None for item in observations
        ),
        "next_coordinate_decode_token_rows": next_rows,
        "observed_next_shard_estimate_compressed_bytes": observed_estimate,
        "frozen_minimum_one_shard_headroom_bytes": FROZEN_MINIMUM_ONE_SHARD_HEADROOM_BYTES,
        "required_headroom_bytes": required,
        "filesystem_available_bytes": filesystem_available_bytes,
        "sufficient": filesystem_available_bytes >= required,
        "limitations": NEXT_LAUNCH_HEADROOM_LIMITATIONS,
    }


def _next_launch_headroom_preflight(
    *,
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    coordinate: ShardCoordinate,
) -> dict[str, Any]:
    payload = _current_headroom_signal(
        output_root=output_root,
        records=records,
        coordinate=coordinate,
    )
    if not payload["sufficient"]:
        raise InfrastructurePauseSignal(payload)
    return payload


def _current_headroom_signal(
    *,
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    coordinate: ShardCoordinate,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    filesystem = os.statvfs(output_root)
    available = filesystem.f_bavail * filesystem.f_frsize
    return _headroom_signal_payload(
        records=records,
        coordinate=coordinate,
        filesystem_available_bytes=available,
    )


def _matrix_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    prerequisites: FrozenPrerequisites,
    evaluator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    worker_count: int = 1,
    gpu_worker_leases: Mapping[int, Mapping[str, Any]],
    worker_ledger_root: Mapping[str, Any] | None = None,
    worker_ledger_bindings: Sequence[Mapping[str, Any]] = (),
    globally_committed_shards: int | None = None,
    storage_records: Sequence[Mapping[str, Any]] | None = None,
    storage_remaining_coordinates: Sequence[ShardCoordinate] | None = None,
) -> dict[str, Any]:
    _require(type(worker_count) is int and worker_count > 0, "Matrix worker count is invalid.")
    committed_shards = (
        len(records) if globally_committed_shards is None else globally_committed_shards
    )
    _require(
        type(committed_shards) is int
        and len(records) <= committed_shards <= EXPECTED_SHARDS
        and (worker_count > 1 or committed_shards == len(records)),
        "Matrix global committed-shard count is invalid.",
    )
    terminal = len(records) == EXPECTED_SHARDS
    pass_count = sum(record.get("integrity_decision") == "INTEGRITY-PASS" for record in records)
    fail_count = sum(record.get("integrity_decision") == "INTEGRITY-FAIL" for record in records)
    integrity_status: str | None = None
    if terminal:
        integrity_status = "INTEGRITY-PASS" if fail_count == 0 else "INTEGRITY-FAIL"
    gpu_registry = _gpu_registry_payload(gpu_worker_leases, worker_count=worker_count)
    gpu_device_registry = _gpu_device_identity_registry(
        gpu_worker_leases, worker_count=worker_count
    )
    frozen_projection = prerequisites.public_binding.get("execution_environment_projection")
    _require(
        isinstance(frozen_projection, Mapping)
        and isinstance(frozen_projection.get("selected_device_class"), Mapping),
        "Frozen selected GPU device class is missing from prerequisites.",
    )
    frozen_projection_map = cast(Mapping[str, Any], frozen_projection)
    selected_device_class = dict(
        cast(Mapping[str, Any], frozen_projection_map["selected_device_class"])
    )
    _require(
        all(item.get("selected_device_class") == selected_device_class for item in gpu_registry),
        "GPU worker leases do not match the frozen selected-device class.",
    )
    _require(
        (worker_count == 1 and tuple(gpu_worker_leases) == (0,))
        or (worker_count > 1 and (not terminal or len(gpu_worker_leases) == worker_count)),
        "Matrix GPU worker-lease registry is incomplete.",
    )
    if worker_count == 1:
        _require(
            worker_ledger_root is None and not worker_ledger_bindings,
            "Single-worker matrix requires an exact empty worker-ledger registry.",
        )
    else:
        raw_worker_root = {} if worker_ledger_root is None else dict(worker_ledger_root)
        raw_root_path = raw_worker_root.get("path")
        _require(
            set(raw_worker_root) == {"path", "semantics", "path_semantics", "worker_count"}
            and isinstance(raw_root_path, str)
            and Path(raw_root_path) == Path(os.path.abspath(raw_root_path))
            and raw_worker_root.get("semantics") == WORKER_LEDGER_ROOT_SEMANTICS
            and raw_worker_root.get("path_semantics") == WORKER_LEDGER_PATH_SEMANTICS
            and raw_worker_root.get("worker_count") == worker_count,
            "Distributed worker-ledger root contract drifted.",
        )
    observed_storage_records = records if storage_records is None else storage_records
    _require(
        len(observed_storage_records) == committed_shards,
        "Storage observations do not cover every globally committed shard.",
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": "terminal" if terminal else "in_progress",
        "integrity_status": integrity_status,
        "source": prerequisites.context.source,
        "manifest": prerequisites.context.manifest_binding,
        "attestation_contract": prerequisites.context.manifest_binding["attestation"],
        "prerequisites": dict(prerequisites.public_binding),
        "canonical_evaluator": dict(evaluator_binding),
        "matrix_lock": dict(matrix_lock_binding),
        "gpu_worker_leases": gpu_registry,
        "gpu_device_identity_registry": gpu_device_registry,
        "selected_device_class": selected_device_class,
        "worker_ledger_assignment_rule": (None if worker_count == 1 else WORKER_ASSIGNMENT_RULE),
        "worker_ledger_root": None if worker_ledger_root is None else dict(worker_ledger_root),
        "worker_ledger_registry": [dict(item) for item in worker_ledger_bindings],
        "execution_mode": "single-worker" if worker_count == 1 else "distributed-workers",
        "worker_count": worker_count,
        "cell_claim_semantics": CELL_CLAIM_SEMANTICS,
        "crash_recovery_boundary": CRASH_RECOVERY_BOUNDARY,
        "outcome_policy": OUTCOME_POLICY,
        "outcome_dependent_early_stopping": False,
        "outcome_selection_performed": False,
        "quality_outcomes_aggregated": False,
        "coordinate_order": ("scale,training_seed,budget,family,context,replicate"),
        "coordinate_count": EXPECTED_SHARDS,
        "coordinate_digest": coordinate_digest(),
        "examples_per_shard": EXAMPLES_PER_SHARD,
        "arm_names": list(ARM_NAMES),
        "expected_outcome_rows": EXPECTED_OUTCOME_ROWS,
        "completed_shards": committed_shards,
        "canonical_prefix_shards": len(records),
        "globally_committed_shards": committed_shards,
        "integrity_pass_shards": pass_count,
        "integrity_fail_shards": fail_count,
        "storage_aggregation_scope": "all-globally-committed-records-v1",
        "storage": _storage_rollup(
            observed_storage_records,
            remaining_coordinates=storage_remaining_coordinates,
        ),
        "records": list(records),
    }
    return _attested_payload(payload, trust_root=prerequisites.trust_root)


def _paused_matrix_payload(
    resume_payload: Mapping[str, Any],
    *,
    signal: Mapping[str, Any],
    worker_index: int,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _verify_attested_payload(resume_payload, trust_root=trust_root)
    _require(
        set(resume_payload) == _MATRIX_FIELDS and resume_payload.get("status") == "in_progress",
        "Infrastructure pause requires a valid in-progress resume payload.",
    )
    _require(
        set(signal) == _HEADROOM_SIGNAL_FIELDS
        and signal.get("schema_version") == 1
        and signal.get("signal_type") == "infrastructure-storage-headroom-pause-v1"
        and signal.get("retryable") is True
        and signal.get("method") == NEXT_LAUNCH_HEADROOM_METHOD
        and signal.get("outcome_independent") is True
        and signal.get("sufficient") is False
        and type(signal.get("filesystem_available_bytes")) is int
        and type(signal.get("required_headroom_bytes")) is int
        and cast(int, signal["filesystem_available_bytes"])
        < cast(int, signal["required_headroom_bytes"]),
        "Infrastructure pause signal is not an exact insufficient-headroom reason.",
    )
    worker_count = resume_payload.get("worker_count")
    _require(
        type(worker_count) is int and 0 <= worker_index < worker_count,
        "Infrastructure pause worker identity is invalid.",
    )
    body = dict(resume_payload)
    body.pop("payload_sha256")
    body.pop("attestation")
    body.pop("integrity_status")
    body.pop("integrity_pass_shards")
    body.pop("integrity_fail_shards")
    body["status"] = "paused_infrastructure"
    body["infrastructure_pause"] = {
        **dict(signal),
        "resume_rule": INFRASTRUCTURE_PAUSE_RESUME_RULE,
        "matrix_records_digest": contract.json_digest(resume_payload["records"]),
        "canonical_prefix_shards": resume_payload["canonical_prefix_shards"],
        "globally_committed_shards": resume_payload["globally_committed_shards"],
        "worker_ledger_registry_digest": contract.json_digest(
            resume_payload["worker_ledger_registry"]
        ),
        "resume_matrix_payload_sha256": resume_payload["payload_sha256"],
        "pause_worker_index": worker_index,
        "pause_worker_count": worker_count,
    }
    return _attested_payload(body, trust_root=trust_root)


def _draining_matrix_payload(
    resume_payload: Mapping[str, Any],
    *,
    signal: Mapping[str, Any],
    worker_index: int,
    active_claim_coordinate_indices: Sequence[int],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _verify_attested_payload(resume_payload, trust_root=trust_root)
    _require(
        set(resume_payload) == _MATRIX_FIELDS and resume_payload.get("status") == "in_progress",
        "Infrastructure drain requires a valid in-progress resume payload.",
    )
    _require(
        set(signal) == _HEADROOM_SIGNAL_FIELDS
        and signal.get("schema_version") == 1
        and signal.get("signal_type") == "infrastructure-storage-headroom-pause-v1"
        and signal.get("retryable") is True
        and signal.get("method") == NEXT_LAUNCH_HEADROOM_METHOD
        and signal.get("outcome_independent") is True
        and type(signal.get("sufficient")) is bool,
        "Infrastructure drain headroom signal is invalid.",
    )
    worker_count = resume_payload.get("worker_count")
    _require(
        type(worker_count) is int and 0 <= worker_index < worker_count,
        "Infrastructure drain owner identity is invalid.",
    )
    active = sorted(active_claim_coordinate_indices)
    _require(
        len(active) == len(set(active))
        and all(type(index) is int and 0 <= index < EXPECTED_SHARDS for index in active),
        "Infrastructure drain active-claim inventory is invalid.",
    )
    body = dict(resume_payload)
    body.pop("payload_sha256")
    body.pop("attestation")
    body.pop("integrity_status")
    body.pop("integrity_pass_shards")
    body.pop("integrity_fail_shards")
    body["status"] = "draining_infrastructure"
    body["infrastructure_drain"] = {
        **dict(signal),
        "resume_rule": INFRASTRUCTURE_PAUSE_RESUME_RULE,
        "matrix_records_digest": contract.json_digest(resume_payload["records"]),
        "canonical_prefix_shards": resume_payload["canonical_prefix_shards"],
        "globally_committed_shards": resume_payload["globally_committed_shards"],
        "worker_ledger_registry_digest": contract.json_digest(
            resume_payload["worker_ledger_registry"]
        ),
        "resume_matrix_payload_sha256": resume_payload["payload_sha256"],
        "pause_worker_index": worker_index,
        "pause_worker_count": worker_count,
        "drain_rule": INFRASTRUCTURE_DRAIN_RULE,
        "active_claim_coordinate_indices": active,
        "active_claim_count": len(active),
    }
    return _attested_payload(body, trust_root=trust_root)


def _infrastructure_owner(payload: Mapping[str, Any]) -> int:
    status = payload.get("status")
    _require(
        status in {"paused_infrastructure", "draining_infrastructure"},
        "Matrix is not infrastructure-blocked.",
    )
    reason_field = (
        "infrastructure_pause" if status == "paused_infrastructure" else "infrastructure_drain"
    )
    reason = payload.get(reason_field)
    _require(isinstance(reason, Mapping), "Infrastructure owner binding is missing.")
    owner = cast(Mapping[str, Any], reason).get("pause_worker_index")
    _require(type(owner) is int, "Infrastructure owner binding is invalid.")
    return cast(int, owner)


def _rebind_infrastructure_after_commit(
    *,
    prior_blocked_payload: Mapping[str, Any],
    resume_payload: Mapping[str, Any],
    ledgers: Mapping[int, Sequence[Mapping[str, Any]]],
    merged_records: Mapping[int, Mapping[str, Any]],
    active_claim_coordinate_indices: Sequence[int],
    output_root: Path,
    worker_count: int,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    owner = _infrastructure_owner(prior_blocked_payload)
    owner_records = ledgers.get(owner)
    _require(owner_records is not None, "Infrastructure owner worker ledger disappeared.")
    bound_owner_records = cast(Sequence[Mapping[str, Any]], owner_records)
    assigned = _assigned_coordinates(worker_index=owner, worker_count=worker_count)
    _require(
        len(bound_owner_records) < len(assigned),
        "Infrastructure owner no longer has a resumable coordinate.",
    )
    signal = _current_headroom_signal(
        output_root=output_root,
        records=[merged_records[index] for index in sorted(merged_records)],
        coordinate=assigned[len(bound_owner_records)],
    )
    if active_claim_coordinate_indices or signal["sufficient"]:
        return _draining_matrix_payload(
            resume_payload,
            signal=signal,
            worker_index=owner,
            active_claim_coordinate_indices=active_claim_coordinate_indices,
            trust_root=trust_root,
        )
    return _paused_matrix_payload(
        resume_payload,
        signal=signal,
        worker_index=owner,
        trust_root=trust_root,
    )


def _headroom_preflight_with_attested_pause(
    *,
    output_root: Path,
    matrix_summary: Path,
    storage_records: Sequence[Mapping[str, Any]],
    coordinate: ShardCoordinate,
    current_payload: Mapping[str, Any],
    resume_payload: Mapping[str, Any],
    worker_index: int,
    active_claim_coordinate_indices: Sequence[int] = (),
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    _require(
        current_payload.get("status")
        in {"in_progress", "paused_infrastructure", "draining_infrastructure"}
        and resume_payload.get("status") == "in_progress",
        "Headroom preflight requires in-progress, draining, or paused infrastructure state.",
    )
    if current_payload.get("status") in {"paused_infrastructure", "draining_infrastructure"}:
        reason_field = (
            "infrastructure_pause"
            if current_payload.get("status") == "paused_infrastructure"
            else "infrastructure_drain"
        )
        prior_pause = current_payload.get(reason_field)
        _require(
            isinstance(prior_pause, Mapping)
            and cast(Mapping[str, Any], prior_pause).get("pause_worker_index") == worker_index
            and cast(Mapping[str, Any], prior_pause).get("resume_matrix_payload_sha256")
            == resume_payload.get("payload_sha256"),
            "Only the bound paused worker may rerun infrastructure headroom.",
        )
    try:
        result = _next_launch_headroom_preflight(
            output_root=output_root,
            records=storage_records,
            coordinate=coordinate,
        )
    except InfrastructurePauseSignal as signal:
        if active_claim_coordinate_indices:
            draining = _draining_matrix_payload(
                resume_payload,
                signal=signal.payload,
                worker_index=worker_index,
                active_claim_coordinate_indices=active_claim_coordinate_indices,
                trust_root=trust_root,
            )
            _atomic_write_json(matrix_summary, draining)
            raise InfrastructurePauseSignal(
                cast(Mapping[str, Any], draining["infrastructure_drain"])
            ) from None
        paused = _paused_matrix_payload(
            resume_payload,
            signal=signal.payload,
            worker_index=worker_index,
            trust_root=trust_root,
        )
        _atomic_write_json(matrix_summary, paused)
        raise InfrastructurePauseSignal(
            cast(Mapping[str, Any], paused["infrastructure_pause"])
        ) from None
    if (
        current_payload.get("status") == "draining_infrastructure"
        and active_claim_coordinate_indices
    ):
        draining = _draining_matrix_payload(
            resume_payload,
            signal=result,
            worker_index=worker_index,
            active_claim_coordinate_indices=active_claim_coordinate_indices,
            trust_root=trust_root,
        )
        _atomic_write_json(matrix_summary, draining)
        raise InfrastructurePauseSignal(cast(Mapping[str, Any], draining["infrastructure_drain"]))
    if current_payload.get("status") in {"paused_infrastructure", "draining_infrastructure"}:
        _atomic_write_json(matrix_summary, resume_payload)
    return result


def _validate_record_shape(record: Mapping[str, Any], coordinate: ShardCoordinate) -> None:
    for field, expected in coordinate.payload.items():
        _require(record.get(field) == expected, f"Matrix record {field} drifted: {coordinate.key}.")
    _require(record.get("coordinate_key") == coordinate.key, "Matrix coordinate key drifted.")
    _require(record.get("status") == "terminal", "Matrix record is not terminal.")
    _require(contract.is_sha256(record.get("launch_nonce")), "Matrix launch nonce is invalid.")


def _validate_record_against_coordinate(
    record: Mapping[str, Any],
    *,
    coordinate: ShardCoordinate,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    evaluator_script: Path,
    gpu_lease_binding: Mapping[str, Any],
    verify_bundle: bool,
) -> dict[str, Any]:
    _validate_record_shape(record, coordinate)
    inputs = prerequisites.bundles[(coordinate.scale, coordinate.training_seed, coordinate.budget)]
    launch_nonce = cast(str, record["launch_nonce"])
    command = build_evaluator_command(
        evaluator_script=evaluator_script,
        envelope_path=shard_envelope_path(output_root, coordinate),
        manifest_path=prerequisites.context.manifest_path,
        coordinate=coordinate,
        inputs=inputs,
        launch_nonce=launch_nonce,
        gpu_lease_binding=gpu_lease_binding,
    )
    if verify_bundle:
        envelope_path = shard_envelope_path(output_root, coordinate)
        loaded = load_and_validate_shard_bundle(
            envelope_path,
            coordinate=coordinate,
            inputs=inputs,
            launch_nonce=launch_nonce,
            trust_root=prerequisites.trust_root,
            gpu_lease_binding=gpu_lease_binding,
            execution_environment_projection=cast(
                Mapping[str, Any],
                prerequisites.public_binding["execution_environment_projection"],
            ),
        )
        expected = _run_record(
            loaded,
            coordinate=coordinate,
            envelope_path=envelope_path,
            inputs=inputs,
            launch_nonce=launch_nonce,
            command=command,
        )
        _require(record == expected, f"Controller matrix record drifted: {coordinate.key}.")
    decision = record.get("integrity_decision")
    _require(
        decision in {"INTEGRITY-PASS", "INTEGRITY-FAIL"},
        "Matrix record integrity decision is invalid.",
    )
    return dict(record)


def validate_matrix_summary(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    evaluator_script: Path,
    evaluator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    verify_bundles: bool = True,
    expected_worker_count: int | None = None,
    expected_gpu_worker_leases: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    _verify_attested_payload(payload, trust_root=prerequisites.trust_root)
    paused_infrastructure = payload.get("status") == "paused_infrastructure"
    draining_infrastructure = payload.get("status") == "draining_infrastructure"
    infrastructure_blocked = paused_infrastructure or draining_infrastructure
    expected_schema = (
        _PAUSED_MATRIX_FIELDS
        if paused_infrastructure
        else _DRAINING_MATRIX_FIELDS
        if draining_infrastructure
        else _MATRIX_FIELDS
    )
    _require(set(payload) == expected_schema, "Controller matrix top-level schema drifted.")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Controller matrix schema drifted.")
    _require(payload.get("experiment_id") == EXPERIMENT_ID, "Wrong controller matrix ID.")
    _require(payload.get("artifact_type") == ARTIFACT_TYPE, "Controller matrix type drifted.")
    _require(payload.get("source") == prerequisites.context.source, "Matrix source drifted.")
    _require(
        payload.get("manifest") == prerequisites.context.manifest_binding,
        "Matrix manifest binding drifted.",
    )
    _require(
        payload.get("attestation_contract")
        == prerequisites.context.manifest_binding["attestation"],
        "Matrix attestation contract drifted.",
    )
    _require(
        payload.get("prerequisites") == dict(prerequisites.public_binding),
        "Matrix prerequisite binding drifted.",
    )
    _require(
        payload.get("canonical_evaluator") == dict(evaluator_binding),
        "Matrix evaluator snapshot drifted.",
    )
    _require(
        payload.get("matrix_lock") == dict(matrix_lock_binding),
        "Matrix lock binding drifted.",
    )
    worker_count = payload.get("worker_count")
    _require(
        type(worker_count) is int
        and worker_count > 0
        and payload.get("execution_mode")
        == ("single-worker" if worker_count == 1 else "distributed-workers")
        and (expected_worker_count is None or worker_count == expected_worker_count),
        "Matrix worker execution contract drifted.",
    )
    raw_gpu_registry = payload.get("gpu_worker_leases")
    _require(isinstance(raw_gpu_registry, list), "Matrix GPU lease registry is invalid.")
    gpu_registry_list = cast(list[Mapping[str, Any]], raw_gpu_registry)
    _require(
        all(
            isinstance(item, Mapping) and type(item.get("worker_index")) is int
            for item in gpu_registry_list
        ),
        "Matrix GPU lease registry entry is invalid.",
    )
    observed_gpu_bindings = {
        cast(int, item["worker_index"]): {
            key: value for key, value in item.items() if key != "worker_index"
        }
        for item in gpu_registry_list
    }
    _require(
        gpu_registry_list
        == _gpu_registry_payload(observed_gpu_bindings, worker_count=cast(int, worker_count))
        and (
            expected_gpu_worker_leases is None
            or gpu_registry_list
            == _gpu_registry_payload(
                expected_gpu_worker_leases, worker_count=cast(int, worker_count)
            )
        ),
        "Matrix GPU worker-lease registry drifted.",
    )
    frozen_projection = prerequisites.public_binding.get("execution_environment_projection")
    _require(
        isinstance(frozen_projection, Mapping)
        and isinstance(frozen_projection.get("selected_device_class"), Mapping)
        and payload.get("selected_device_class") == frozen_projection.get("selected_device_class")
        and payload.get("gpu_device_identity_registry")
        == _gpu_device_identity_registry(
            observed_gpu_bindings, worker_count=cast(int, worker_count)
        ),
        "Matrix selected-device class or physical-identity registry drifted.",
    )
    raw_worker_registry = payload.get("worker_ledger_registry")
    _require(isinstance(raw_worker_registry, list), "Matrix worker-ledger registry is invalid.")
    replayed_worker_ledgers: dict[int, list[dict[str, Any]]] = {}
    replayed_prefix: list[dict[str, Any]] = []
    replayed_merged: dict[int, dict[str, Any]] = {}
    if cast(int, worker_count) == 1:
        _require(
            payload.get("worker_ledger_assignment_rule") is None
            and payload.get("worker_ledger_root") is None
            and raw_worker_registry == []
            and not _worker_ledger_root(output_root).exists(),
            "Single-worker matrix worker-ledger registry must be exact empty.",
        )
    else:
        expected_root_binding = _worker_ledger_root_binding(
            output_root, worker_count=cast(int, worker_count)
        )
        _require(
            payload.get("worker_ledger_assignment_rule") == WORKER_ASSIGNMENT_RULE
            and payload.get("worker_ledger_root") == expected_root_binding,
            "Distributed worker-ledger assignment or root binding drifted.",
        )
        replayed_worker_ledgers, replayed_gpu_bindings = _load_worker_ledgers(
            output_root=output_root,
            worker_count=cast(int, worker_count),
            prerequisites=prerequisites,
            evaluator_script=evaluator_script,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=matrix_lock_binding,
            verify_bundles=verify_bundles,
        )
        expected_worker_registry = _worker_ledger_registry(
            output_root,
            worker_count=cast(int, worker_count),
            completed_by_worker={
                index: len(worker_records)
                for index, worker_records in replayed_worker_ledgers.items()
            },
        )
        _require(
            raw_worker_registry == expected_worker_registry
            and observed_gpu_bindings == replayed_gpu_bindings,
            "Public matrix worker-ledger registry or GPU provenance drifted.",
        )
        replayed_prefix, replayed_merged = _merge_worker_records(
            replayed_worker_ledgers, worker_count=cast(int, worker_count)
        )
    _require(
        payload.get("coordinate_count") == EXPECTED_SHARDS
        and payload.get("coordinate_digest") == coordinate_digest()
        and payload.get("coordinate_order")
        == "scale,training_seed,budget,family,context,replicate",
        "Matrix coordinate inventory drifted.",
    )
    _require(
        payload.get("examples_per_shard") == EXAMPLES_PER_SHARD
        and tuple(payload.get("arm_names", ())) == ARM_NAMES
        and payload.get("expected_outcome_rows") == EXPECTED_OUTCOME_ROWS,
        "Matrix shard/arm cardinality drifted.",
    )
    _require(
        payload.get("outcome_dependent_early_stopping") is False
        and payload.get("outcome_selection_performed") is False
        and payload.get("quality_outcomes_aggregated") is False
        and payload.get("outcome_policy") == OUTCOME_POLICY,
        "Matrix outcome-independence boundary drifted.",
    )
    _require(
        payload.get("cell_claim_semantics") == CELL_CLAIM_SEMANTICS
        and payload.get("crash_recovery_boundary") == CRASH_RECOVERY_BOUNDARY,
        "Matrix claim or crash-recovery contract drifted.",
    )
    records = payload.get("records")
    _require(isinstance(records, list), "Controller matrix records are invalid.")
    records = cast(list[Mapping[str, Any]], records)
    _require(len(records) <= EXPECTED_SHARDS, "Controller matrix contains extra records.")
    globally_committed_shards = payload.get("globally_committed_shards")
    _require(
        type(globally_committed_shards) is int,
        "Globally committed shard count is invalid.",
    )
    global_count = cast(int, globally_committed_shards)
    _require(
        payload.get("completed_shards") == global_count
        and payload.get("canonical_prefix_shards") == len(records)
        and len(records) <= global_count <= EXPECTED_SHARDS
        and (cast(int, worker_count) > 1 or global_count == len(records)),
        "Canonical-prefix or globally committed shard count drifted.",
    )
    result: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    canonical = _canonical_evaluator(evaluator_script)
    for coordinate_index, (record, coordinate) in enumerate(
        zip(records, coordinates()[: len(records)], strict=True)
    ):
        _require(isinstance(record, Mapping), "Controller matrix record is invalid.")
        validated_record = _validate_record_against_coordinate(
            record,
            coordinate=coordinate,
            output_root=output_root,
            prerequisites=prerequisites,
            evaluator_script=canonical,
            gpu_lease_binding=observed_gpu_bindings[
                0 if cast(int, worker_count) == 1 else coordinate_index % cast(int, worker_count)
            ],
            verify_bundle=verify_bundles,
        )
        decision = validated_record["integrity_decision"]
        pass_count += decision == "INTEGRITY-PASS"
        fail_count += decision == "INTEGRITY-FAIL"
        result.append(validated_record)
    if cast(int, worker_count) > 1:
        _require(
            result == replayed_prefix and global_count == len(replayed_merged),
            "Public matrix prefix/global counts drifted from bound worker ledgers.",
        )
    terminal = len(records) == EXPECTED_SHARDS
    _require(
        (worker_count == 1 and tuple(observed_gpu_bindings) == (0,))
        or (
            cast(int, worker_count) > 1
            and (not terminal or len(observed_gpu_bindings) == worker_count)
        ),
        "Matrix GPU worker-lease registry is incomplete for its status.",
    )
    if infrastructure_blocked:
        _require(not terminal, "Terminal controller matrix may not be infrastructure-blocked.")
    else:
        _require(
            payload.get("status") == ("terminal" if terminal else "in_progress"),
            "Controller matrix status drifted.",
        )
    expected_integrity: str | None = None
    if terminal:
        expected_integrity = "INTEGRITY-PASS" if fail_count == 0 else "INTEGRITY-FAIL"
    if infrastructure_blocked:
        reason_field = "infrastructure_pause" if paused_infrastructure else "infrastructure_drain"
        pause = payload.get(reason_field)
        reason_schema = (
            _INFRASTRUCTURE_PAUSE_FIELDS if paused_infrastructure else _INFRASTRUCTURE_DRAIN_FIELDS
        )
        _require(
            isinstance(pause, Mapping) and set(pause) == reason_schema,
            "Infrastructure pause/drain reason schema drifted.",
        )
        pause_map = cast(Mapping[str, Any], pause)
        pause_worker_index = pause_map.get("pause_worker_index")
        _require(
            pause_map.get("schema_version") == 1
            and pause_map.get("signal_type") == "infrastructure-storage-headroom-pause-v1"
            and pause_map.get("retryable") is True
            and pause_map.get("method") == NEXT_LAUNCH_HEADROOM_METHOD
            and pause_map.get("outcome_independent") is True
            and (
                pause_map.get("sufficient") is False
                if paused_infrastructure
                else type(pause_map.get("sufficient")) is bool
            )
            and pause_map.get("limitations") == NEXT_LAUNCH_HEADROOM_LIMITATIONS
            and pause_map.get("resume_rule") == INFRASTRUCTURE_PAUSE_RESUME_RULE
            and type(pause_map.get("filesystem_available_bytes")) is int
            and type(pause_map.get("required_headroom_bytes")) is int
            and pause_map.get("sufficient")
            is (
                cast(int, pause_map["filesystem_available_bytes"])
                >= cast(int, pause_map["required_headroom_bytes"])
            )
            and pause_map.get("completed_record_count") == global_count
            and pause_map.get("canonical_prefix_shards") == len(records)
            and pause_map.get("globally_committed_shards") == global_count
            and pause_map.get("matrix_records_digest") == contract.json_digest(records)
            and pause_map.get("worker_ledger_registry_digest")
            == contract.json_digest(payload["worker_ledger_registry"])
            and pause_map.get("pause_worker_count") == worker_count
            and type(pause_worker_index) is int
            and 0 <= pause_worker_index < cast(int, worker_count),
            "Infrastructure pause reason or bound progress drifted.",
        )
        if draining_infrastructure:
            raw_active_claims = pause_map.get("active_claim_coordinate_indices")
            live_claim_indices: list[int] = []
            for claim_index, claim_coordinate in enumerate(coordinates()):
                claim_path = shard_output_dir(output_root, claim_coordinate) / CELL_CLAIM_NAME
                if claim_path.exists():
                    _validated_live_claim(
                        claim_path,
                        coordinate=claim_coordinate,
                        canonical_index=claim_index,
                        worker_count=cast(int, worker_count),
                    )
                    live_claim_indices.append(claim_index)
            _require(
                cast(int, worker_count) > 1
                and pause_map.get("drain_rule") == INFRASTRUCTURE_DRAIN_RULE
                and isinstance(raw_active_claims, list)
                and pause_map.get("active_claim_count") == len(raw_active_claims)
                and raw_active_claims == sorted(set(raw_active_claims))
                and raw_active_claims == live_claim_indices
                and all(
                    type(index) is int
                    and 0 <= index < EXPECTED_SHARDS
                    and index not in replayed_merged
                    for index in raw_active_claims
                ),
                "Infrastructure drain active-claim binding drifted.",
            )
        else:
            _require(
                not any(
                    (shard_output_dir(output_root, claim_coordinate) / CELL_CLAIM_NAME).exists()
                    for claim_coordinate in coordinates()
                ),
                "Infrastructure pause may not coexist with a live distributed claim.",
            )
        if cast(int, worker_count) == 1:
            expected_pause_coordinate = coordinates()[len(records)]
        else:
            pause_worker_records = replayed_worker_ledgers.get(cast(int, pause_worker_index))
            _require(
                pause_worker_records is not None,
                "Infrastructure pause worker has no bound worker ledger.",
            )
            pause_assignment = _assigned_coordinates(
                worker_index=cast(int, pause_worker_index),
                worker_count=cast(int, worker_count),
            )
            pause_record_count = len(cast(list[dict[str, Any]], pause_worker_records))
            _require(
                pause_record_count < len(pause_assignment),
                "Infrastructure pause worker has no remaining coordinate.",
            )
            expected_pause_coordinate = pause_assignment[pause_record_count]
        pause_storage_records = (
            records
            if cast(int, worker_count) == 1
            else [replayed_merged[index] for index in sorted(replayed_merged)]
        )
        expected_signal = _headroom_signal_payload(
            records=pause_storage_records,
            coordinate=expected_pause_coordinate,
            filesystem_available_bytes=cast(int, pause_map["filesystem_available_bytes"]),
        )
        _require(
            {field: pause_map[field] for field in _HEADROOM_SIGNAL_FIELDS} == expected_signal,
            "Infrastructure pause reason does not exactly replay from bound evidence.",
        )
        resume_body = dict(payload)
        resume_body.pop("payload_sha256")
        resume_body.pop("attestation")
        resume_body.pop(reason_field)
        resume_body["status"] = "in_progress"
        resume_body["integrity_status"] = None
        resume_body["integrity_pass_shards"] = pass_count
        resume_body["integrity_fail_shards"] = fail_count
        resume_payload = _digest_bound_payload(resume_body)
        _require(
            pause_map.get("resume_matrix_payload_sha256") == resume_payload["payload_sha256"],
            "Infrastructure pause resume-payload binding drifted.",
        )
    else:
        _require(
            payload.get("integrity_status") == expected_integrity,
            "Matrix integrity status drifted.",
        )
        _require(
            payload.get("integrity_pass_shards") == pass_count
            and payload.get("integrity_fail_shards") == fail_count,
            "Matrix integrity counts drifted.",
        )
    storage = payload.get("storage")
    _require(
        payload.get("storage_aggregation_scope") == "all-globally-committed-records-v1"
        and isinstance(storage, Mapping),
        "Matrix storage rollup or aggregation scope drifted.",
    )
    storage_map = cast(Mapping[str, Any], storage)
    if cast(int, worker_count) > 1:
        replayed_storage_records = [replayed_merged[index] for index in sorted(replayed_merged)]
        replayed_remaining = [
            coordinate
            for index, coordinate in enumerate(coordinates())
            if index not in replayed_merged
        ]
        _require(
            dict(storage_map)
            == _storage_rollup(
                replayed_storage_records,
                remaining_coordinates=replayed_remaining,
            ),
            "Sparse matrix storage drifted from bound worker ledgers.",
        )
    elif global_count == len(records):
        _require(
            dict(storage_map) == _storage_rollup(records),
            "Matrix storage rollup drifted from its completed records.",
        )
    return result


def _validate_worker_identity(*, worker_index: int, worker_count: int) -> None:
    _require(
        type(worker_count) is int and 2 <= worker_count <= EXPECTED_SHARDS,
        "Distributed worker count must be between 2 and the frozen shard count.",
    )
    _require(
        type(worker_index) is int and 0 <= worker_index < worker_count,
        "Distributed worker index is outside its worker count.",
    )


def _assigned_coordinates(*, worker_index: int, worker_count: int) -> tuple[ShardCoordinate, ...]:
    _validate_worker_identity(worker_index=worker_index, worker_count=worker_count)
    return tuple(
        coordinate
        for index, coordinate in enumerate(coordinates())
        if index % worker_count == worker_index
    )


def _worker_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    worker_index: int,
    worker_count: int,
    prerequisites: FrozenPrerequisites,
    evaluator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    gpu_lease_binding: Mapping[str, Any],
) -> dict[str, Any]:
    assigned = _assigned_coordinates(worker_index=worker_index, worker_count=worker_count)
    _require(len(records) <= len(assigned), "Worker ledger contains extra shard records.")
    pass_count = sum(record.get("integrity_decision") == "INTEGRITY-PASS" for record in records)
    fail_count = sum(record.get("integrity_decision") == "INTEGRITY-FAIL" for record in records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "artifact_type": WORKER_ARTIFACT_TYPE,
        "status": "terminal" if len(records) == len(assigned) else "in_progress",
        "source": prerequisites.context.source,
        "manifest": prerequisites.context.manifest_binding,
        "attestation_contract": prerequisites.context.manifest_binding["attestation"],
        "prerequisites": dict(prerequisites.public_binding),
        "canonical_evaluator": dict(evaluator_binding),
        "matrix_lock": dict(matrix_lock_binding),
        "gpu_lease": _validate_gpu_lease_binding(gpu_lease_binding, require_live=True),
        "worker_index": worker_index,
        "worker_count": worker_count,
        "assignment_rule": WORKER_ASSIGNMENT_RULE,
        "global_coordinate_count": EXPECTED_SHARDS,
        "global_coordinate_digest": coordinate_digest(),
        "assigned_coordinate_count": len(assigned),
        "assigned_coordinate_digest": contract.json_digest(
            [coordinate.payload for coordinate in assigned]
        ),
        "completed_shards": len(records),
        "integrity_pass_shards": pass_count,
        "integrity_fail_shards": fail_count,
        "outcome_dependent_early_stopping": False,
        "outcome_selection_performed": False,
        "quality_outcomes_aggregated": False,
        "records": list(records),
    }
    return _attested_worker_payload(payload, trust_root=prerequisites.trust_root)


def _validate_worker_ledger(
    payload: Mapping[str, Any],
    *,
    worker_index: int,
    worker_count: int,
    output_root: Path,
    prerequisites: FrozenPrerequisites,
    evaluator_script: Path,
    evaluator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    verify_bundles: bool,
    expected_gpu_lease_binding: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _validate_worker_identity(worker_index=worker_index, worker_count=worker_count)
    _verify_worker_payload(payload, trust_root=prerequisites.trust_root)
    _require(set(payload) == _WORKER_LEDGER_FIELDS, "Worker ledger schema drifted.")
    _require(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("experiment_id") == EXPERIMENT_ID
        and payload.get("artifact_type") == WORKER_ARTIFACT_TYPE,
        "Worker ledger identity drifted.",
    )
    _require(
        payload.get("source") == prerequisites.context.source
        and payload.get("manifest") == prerequisites.context.manifest_binding
        and payload.get("attestation_contract")
        == prerequisites.context.manifest_binding["attestation"]
        and payload.get("prerequisites") == dict(prerequisites.public_binding)
        and payload.get("canonical_evaluator") == dict(evaluator_binding)
        and payload.get("matrix_lock") == dict(matrix_lock_binding),
        "Worker ledger prerequisite/evaluator binding drifted.",
    )
    raw_gpu_lease = payload.get("gpu_lease")
    _require(isinstance(raw_gpu_lease, Mapping), "Worker GPU lease binding is missing.")
    gpu_lease = _validate_gpu_lease_binding(cast(Mapping[str, Any], raw_gpu_lease))
    expected_gpu_lease = (
        None
        if expected_gpu_lease_binding is None
        else _validate_gpu_lease_binding(expected_gpu_lease_binding, require_live=True)
    )
    _require(
        expected_gpu_lease is None or gpu_lease == expected_gpu_lease,
        "Worker GPU lease path or persistent identity drifted on resume.",
    )
    assigned = _assigned_coordinates(worker_index=worker_index, worker_count=worker_count)
    _require(
        payload.get("worker_index") == worker_index
        and payload.get("worker_count") == worker_count
        and payload.get("assignment_rule") == WORKER_ASSIGNMENT_RULE
        and payload.get("global_coordinate_count") == EXPECTED_SHARDS
        and payload.get("global_coordinate_digest") == coordinate_digest()
        and payload.get("assigned_coordinate_count") == len(assigned)
        and payload.get("assigned_coordinate_digest")
        == contract.json_digest([coordinate.payload for coordinate in assigned]),
        "Worker ledger assignment binding drifted.",
    )
    _require(
        payload.get("outcome_dependent_early_stopping") is False
        and payload.get("outcome_selection_performed") is False
        and payload.get("quality_outcomes_aggregated") is False,
        "Worker ledger crossed the outcome-independence boundary.",
    )
    raw_records = payload.get("records")
    _require(isinstance(raw_records, list), "Worker ledger records are invalid.")
    records = cast(list[Mapping[str, Any]], raw_records)
    _require(
        len(records) <= len(assigned) and payload.get("completed_shards") == len(records),
        "Worker ledger completed count drifted.",
    )
    canonical = _canonical_evaluator(evaluator_script)
    result: list[dict[str, Any]] = []
    pass_count = 0
    fail_count = 0
    for record, coordinate in zip(records, assigned[: len(records)], strict=True):
        _require(isinstance(record, Mapping), "Worker ledger record is invalid.")
        validated = _validate_record_against_coordinate(
            record,
            coordinate=coordinate,
            output_root=output_root,
            prerequisites=prerequisites,
            evaluator_script=canonical,
            gpu_lease_binding=gpu_lease,
            verify_bundle=verify_bundles,
        )
        pass_count += validated["integrity_decision"] == "INTEGRITY-PASS"
        fail_count += validated["integrity_decision"] == "INTEGRITY-FAIL"
        result.append(validated)
    _require(
        payload.get("status") == ("terminal" if len(records) == len(assigned) else "in_progress")
        and payload.get("integrity_pass_shards") == pass_count
        and payload.get("integrity_fail_shards") == fail_count,
        "Worker ledger status or integrity counts drifted.",
    )
    return result, gpu_lease


def _load_worker_ledgers(
    *,
    output_root: Path,
    worker_count: int,
    prerequisites: FrozenPrerequisites,
    evaluator_script: Path,
    evaluator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    verify_bundles: bool,
    expected_worker_index: int | None = None,
    expected_gpu_lease_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    _validate_worker_identity(worker_index=0, worker_count=worker_count)
    root = _worker_ledger_root(output_root)
    if not root.exists():
        return {}, {}
    _require(root.is_dir() and not root.is_symlink(), "Worker-ledger root is unsafe.")
    expected = {
        _worker_ledger_path(
            output_root, worker_index=worker_index, worker_count=worker_count
        ): worker_index
        for worker_index in range(worker_count)
    }
    for item in root.iterdir():
        _require(
            not item.is_symlink() and item.is_file() and _absolute(item) in expected,
            f"Worker-ledger root contains an unregistered file: {item}",
        )
    result: dict[int, list[dict[str, Any]]] = {}
    gpu_bindings: dict[int, dict[str, Any]] = {}
    for path, worker_index in expected.items():
        if not path.exists():
            continue
        records, gpu_binding = _validate_worker_ledger(
            _load_json_nofollow(path, label="controller worker ledger"),
            worker_index=worker_index,
            worker_count=worker_count,
            output_root=output_root,
            prerequisites=prerequisites,
            evaluator_script=evaluator_script,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=matrix_lock_binding,
            verify_bundles=verify_bundles,
            expected_gpu_lease_binding=(
                expected_gpu_lease_binding if expected_worker_index == worker_index else None
            ),
        )
        result[worker_index] = records
        gpu_bindings[worker_index] = gpu_binding
    return result, gpu_bindings


def _merge_worker_records(
    ledgers: Mapping[int, Sequence[Mapping[str, Any]]], *, worker_count: int
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    coordinate_items = coordinates()
    coordinate_index = {coordinate.key: index for index, coordinate in enumerate(coordinate_items)}
    merged: dict[int, dict[str, Any]] = {}
    for worker_index, records in ledgers.items():
        _validate_worker_identity(worker_index=worker_index, worker_count=worker_count)
        for record in records:
            key = cast(str, record.get("coordinate_key"))
            _require(key in coordinate_index, "Worker ledger contains an unknown coordinate.")
            index = coordinate_index[key]
            _require(
                index % worker_count == worker_index and index not in merged,
                "Worker ledgers overlap or violate their partition.",
            )
            merged[index] = dict(record)
    prefix: list[dict[str, Any]] = []
    for index in range(len(coordinate_items)):
        merged_record = merged.get(index)
        if merged_record is None:
            break
        prefix.append(merged_record)
    return prefix, merged


def _preflight_output_tree(
    *, output_root: Path, matrix_summary: Path, completed_shards: int
) -> None:
    root = _absolute(output_root)
    summary = _absolute(matrix_summary)
    if not root.exists():
        _require(completed_shards == 0, "Completed matrix prefix lost its output root.")
        return
    _require(root.is_dir() and not root.is_symlink(), "Controller output root is unsafe.")
    allowed: set[Path] = {summary}
    for index, coordinate in enumerate(coordinates()):
        output_dir = shard_output_dir(root, coordinate)
        cursor = output_dir
        while cursor != root:
            allowed.add(cursor)
            cursor = cursor.parent
        bundle = canonical_bundle_paths(root, coordinate)
        if index < completed_shards:
            allowed.update(bundle.values())
            _require(
                all(path.is_file() and not path.is_symlink() for path in bundle.values()),
                f"Completed evaluator bundle is missing or unsafe: {coordinate.key}.",
            )
            _require(
                not (output_dir / CELL_CLAIM_NAME).exists(),
                f"Completed evaluator bundle retained a claim: {coordinate.key}.",
            )
        else:
            _require(
                not any(path.exists() for path in bundle.values())
                and not (output_dir / CELL_CLAIM_NAME).exists(),
                f"Orphan or partial evaluator bundle blocks resume: {coordinate.key}.",
            )
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"Controller output tree contains a symlink: {item}")
        _require(
            _absolute(item) in allowed,
            f"Controller output tree contains an unregistered orphan: {item}",
        )


def _validated_live_claim(
    claim_path: Path,
    *,
    coordinate: ShardCoordinate,
    canonical_index: int,
    worker_count: int,
) -> dict[str, Any]:
    metadata = os.stat(claim_path, follow_symlinks=False)
    _require(
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and metadata.st_nlink == 1
        and stat.S_IMODE(metadata.st_mode) == 0o600,
        f"Distributed cell claim metadata is unsafe: {coordinate.key}.",
    )
    payload = _load_json_nofollow(claim_path, label="distributed cell claim")
    _require(
        set(payload)
        == {
            "schema_version",
            "semantics",
            "coordinate",
            "launch_nonce",
            "pid",
            "process_start_ticks",
            "worker_index",
            "worker_count",
            "created_time_ns",
        },
        f"Distributed cell claim schema drifted: {coordinate.key}.",
    )
    worker_index = canonical_index % worker_count
    pid = payload.get("pid")
    start_ticks = payload.get("process_start_ticks")
    _require(
        payload.get("schema_version") == 1
        and payload.get("semantics") == CELL_CLAIM_SEMANTICS
        and payload.get("coordinate") == coordinate.payload
        and contract.is_sha256(payload.get("launch_nonce"))
        and type(pid) is int
        and pid > 0
        and type(start_ticks) is int
        and start_ticks > 0
        and payload.get("worker_index") == worker_index
        and payload.get("worker_count") == worker_count
        and type(payload.get("created_time_ns")) is int
        and cast(int, payload["created_time_ns"]) > 0,
        f"Distributed cell claim binding drifted: {coordinate.key}.",
    )
    try:
        os.kill(cast(int, pid), 0)
    except (OSError, ProcessLookupError) as error:
        raise ValueError(
            f"Stale distributed cell claim blocks resume: {coordinate.key}."
        ) from error
    _require(
        _process_start_ticks(cast(int, pid)) == start_ticks,
        f"Reused PID in distributed cell claim blocks resume: {coordinate.key}.",
    )
    return payload


def _live_distributed_claims(*, output_root: Path, worker_count: int) -> dict[int, dict[str, Any]]:
    active: dict[int, dict[str, Any]] = {}
    for index, coordinate in enumerate(coordinates()):
        claim_path = shard_output_dir(output_root, coordinate) / CELL_CLAIM_NAME
        if claim_path.exists():
            active[index] = _validated_live_claim(
                claim_path,
                coordinate=coordinate,
                canonical_index=index,
                worker_count=worker_count,
            )
    return active


def _is_evaluator_temporary(path: Path, bundle_paths: Mapping[str, Path]) -> bool:
    return any(
        path.name.startswith(f".{member.name}.") and path.name.endswith(".tmp")
        for member in bundle_paths.values()
    )


def _preflight_distributed_output_tree(
    *,
    output_root: Path,
    matrix_summary: Path,
    merged_records: Mapping[int, Mapping[str, Any]],
    worker_count: int,
) -> dict[int, dict[str, Any]]:
    root = _absolute(output_root)
    summary = _absolute(matrix_summary)
    active_claims: dict[int, dict[str, Any]] = {}
    if not root.exists():
        _require(not merged_records, "Distributed ledgers lost their controller output root.")
        return active_claims
    _require(root.is_dir() and not root.is_symlink(), "Controller output root is unsafe.")
    allowed: set[Path] = {summary}
    for index, coordinate in enumerate(coordinates()):
        output_dir = shard_output_dir(root, coordinate)
        cursor = output_dir
        while cursor != root:
            allowed.add(cursor)
            cursor = cursor.parent
        bundle = canonical_bundle_paths(root, coordinate)
        claim_path = output_dir / CELL_CLAIM_NAME
        record = merged_records.get(index)
        if record is not None:
            allowed.update(bundle.values())
            _require(
                all(path.is_file() and not path.is_symlink() for path in bundle.values()),
                f"Completed distributed evaluator bundle is missing: {coordinate.key}.",
            )
            _require(
                not claim_path.exists(),
                f"Completed distributed evaluator bundle retained a claim: {coordinate.key}.",
            )
            continue
        if claim_path.exists():
            active_claims[index] = _validated_live_claim(
                claim_path,
                coordinate=coordinate,
                canonical_index=index,
                worker_count=worker_count,
            )
            allowed.add(claim_path)
            for path in bundle.values():
                if path.exists():
                    _require(
                        path.is_file() and not path.is_symlink(),
                        f"Active distributed bundle member is unsafe: {coordinate.key}.",
                    )
                    allowed.add(path)
            if output_dir.exists():
                for path in output_dir.iterdir():
                    if _absolute(path) in allowed:
                        continue
                    _require(
                        path.is_file()
                        and not path.is_symlink()
                        and _is_evaluator_temporary(path, bundle),
                        f"Active distributed cell contains an unknown file: {path}",
                    )
                    allowed.add(path)
        else:
            _require(
                not any(path.exists() for path in bundle.values()),
                f"Orphan distributed evaluator bundle blocks resume: {coordinate.key}.",
            )
            if output_dir.exists():
                _require(
                    not any(output_dir.iterdir()),
                    f"Unclaimed distributed cell contains orphan evidence: {coordinate.key}.",
                )
    for item in root.rglob("*"):
        _require(not item.is_symlink(), f"Controller output tree contains a symlink: {item}")
        _require(
            _absolute(item) in allowed,
            f"Controller output tree contains an unregistered orphan: {item}",
        )
    return active_claims


def _assert_empty_bundle(output_root: Path, coordinate: ShardCoordinate) -> None:
    bundle = canonical_bundle_paths(output_root, coordinate)
    _require(
        not any(path.exists() for path in bundle.values()),
        f"Refusing pre-existing evaluator bundle: {coordinate.key}.",
    )
    output_dir = shard_output_dir(output_root, coordinate)
    if output_dir.exists():
        _require(
            not any(output_dir.iterdir()), f"Refusing non-empty shard directory: {coordinate.key}."
        )


def _distributed_layout_guard(layout: MatrixLayout, *, attestation_key_path: Path | None) -> None:
    worker_root = _exact_resolved_path(
        _worker_ledger_root(layout.output_root), label="Controller worker-ledger root"
    )
    protected = (
        layout.output_root,
        layout.matrix_summary,
        layout.lock_path,
        layout.training_output_root,
        layout.calibration_output_root,
        layout.top_p_output_root,
    )
    _require(
        all(not _paths_overlap(worker_root, path) for path in protected),
        "Controller worker-ledger root must be disjoint from artifacts and inputs.",
    )
    if attestation_key_path is not None:
        key_path = _exact_resolved_path(attestation_key_path, label="Attestation key")
        _require(
            not _paths_overlap(worker_root, key_path),
            "Controller worker-ledger root may not contain the attestation key.",
        )


_SELECTED_DEVICE_CONTEXT_FIELDS = frozenset(
    {
        "compatible_environment_projection",
        "selected_device_class",
        "selected_device_routing_identity",
        "selected_device_logical_index",
        "device_argument",
    }
)
_DEVICE_GUARD_BINDING_FIELDS = frozenset(
    {
        "path",
        "semantics",
        "implementation_path",
        "path_derivation",
        "nonblocking",
        "persistent_inode",
        "device",
        "inode",
    }
)


def _capture_selected_device_context(scheduler_lease: GPULockLease) -> dict[str, Any]:
    scheduler_lease.assert_held()
    captured = execution_environment.capture_execution_environment()
    scheduler_lease.assert_held()
    logical_index = captured.get("current_device_index")
    _require(type(logical_index) is int and logical_index >= 0, "Selected GPU index is invalid.")
    return {
        "compatible_environment_projection": (
            execution_environment.controller_compatible_environment_projection(captured)
        ),
        "selected_device_class": execution_environment.selected_device_class(captured),
        "selected_device_routing_identity": (
            execution_environment.selected_device_routing_identity(captured)
        ),
        "selected_device_logical_index": logical_index,
        "device_argument": f"cuda:{logical_index}",
    }


def _validate_selected_device_context(
    value: Mapping[str, Any], *, prerequisites: FrozenPrerequisites
) -> dict[str, Any]:
    _require(
        set(value) == _SELECTED_DEVICE_CONTEXT_FIELDS,
        "Selected GPU device context schema drifted.",
    )
    projection = value.get("compatible_environment_projection")
    selected_class = value.get("selected_device_class")
    routing_identity = value.get("selected_device_routing_identity")
    logical_index = value.get("selected_device_logical_index")
    device_argument = value.get("device_argument")
    frozen_projection = prerequisites.public_binding.get("execution_environment_projection")
    _require(
        isinstance(projection, Mapping)
        and isinstance(selected_class, Mapping)
        and isinstance(routing_identity, Mapping)
        and isinstance(frozen_projection, Mapping)
        and dict(cast(Mapping[str, Any], projection))
        == dict(cast(Mapping[str, Any], frozen_projection))
        and dict(cast(Mapping[str, Any], selected_class))
        == cast(Mapping[str, Any], projection).get("selected_device_class"),
        "Selected GPU device class or compatible environment projection drifted.",
    )
    identity = dict(cast(Mapping[str, Any], routing_identity))
    _require(
        type(logical_index) is int
        and logical_index >= 0
        and device_argument == f"cuda:{logical_index}",
        "Selected GPU logical index or explicit device argument drifted.",
    )
    try:
        canonical_device_guard_path(identity)
    except RuntimeError as error:
        raise ValueError("Selected GPU routing identity is invalid.") from error
    return {
        "compatible_environment_projection": dict(cast(Mapping[str, Any], projection)),
        "selected_device_class": dict(cast(Mapping[str, Any], selected_class)),
        "selected_device_routing_identity": identity,
        "selected_device_logical_index": logical_index,
        "device_argument": device_argument,
    }


def _acquire_selected_device_guard(
    *, label: str, device_context: Mapping[str, Any], scheduler_lease: GPULockLease
) -> GPULockLease:
    scheduler_lease.assert_held()
    identity = cast(Mapping[str, Any], device_context["selected_device_routing_identity"])
    guard_path = canonical_device_guard_path(identity)
    if scheduler_lease.path == guard_path:
        return scheduler_lease
    guard = acquire_device_guard(label, identity)
    scheduler_lease.assert_held()
    guard.assert_held()
    return guard


def _close_gpu_lease_pair(scheduler_lease: GPULockLease, device_guard_lease: GPULockLease) -> None:
    if device_guard_lease is scheduler_lease:
        scheduler_lease.close()
        return
    failure: BaseException | None = None
    try:
        device_guard_lease.close()
    except BaseException as error:
        failure = error
    finally:
        try:
            scheduler_lease.close()
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


def _gpu_lease_binding(
    scheduler_lease: GPULockLease,
    *,
    device_guard_lease: GPULockLease,
    device_context: Mapping[str, Any],
) -> dict[str, Any]:
    scheduler_lease.assert_held()
    device_guard_lease.assert_held()
    path = _exact_resolved_path(scheduler_lease.path, label="GPU scheduler lease path")
    guard_path = _exact_resolved_path(
        device_guard_lease.path, label="GPU physical-device guard path"
    )
    identity = dict(cast(Mapping[str, Any], device_context["selected_device_routing_identity"]))
    _require(
        guard_path == canonical_device_guard_path(identity),
        "GPU physical-device guard path is not derived from its routing identity.",
    )
    return {
        "path": str(path),
        "semantics": GPU_LEASE_SEMANTICS,
        "implementation_path": GPU_LOCK_IMPLEMENTATION_PATH,
        "nonblocking": True,
        "persistent_inode": True,
        "device": scheduler_lease.device,
        "inode": scheduler_lease.inode,
        "selected_device_class": dict(
            cast(Mapping[str, Any], device_context["selected_device_class"])
        ),
        "selected_device_routing_identity": identity,
        "selected_device_logical_index": device_context["selected_device_logical_index"],
        "device_argument": device_context["device_argument"],
        "device_guard": {
            "path": str(guard_path),
            "semantics": DEVICE_GUARD_SEMANTICS,
            "implementation_path": GPU_LOCK_IMPLEMENTATION_PATH,
            "path_derivation": DEVICE_GUARD_PATH_DERIVATION,
            "nonblocking": True,
            "persistent_inode": True,
            "device": device_guard_lease.device,
            "inode": device_guard_lease.inode,
        },
    }


def _validate_gpu_lease_binding(
    binding: Mapping[str, Any], *, require_live: bool = False
) -> dict[str, Any]:
    _require(
        set(binding)
        == {
            "path",
            "semantics",
            "implementation_path",
            "nonblocking",
            "persistent_inode",
            "device",
            "inode",
            "selected_device_class",
            "selected_device_routing_identity",
            "selected_device_logical_index",
            "device_argument",
            "device_guard",
        },
        "GPU worker-lease binding schema drifted.",
    )
    raw_path = binding.get("path")
    _require(isinstance(raw_path, str) and bool(raw_path), "GPU lease path is invalid.")
    path = Path(cast(str, raw_path))
    _require(
        path == Path(os.path.abspath(path)),
        "GPU lease path is not exact and canonical.",
    )
    _require(
        binding.get("semantics") == GPU_LEASE_SEMANTICS
        and binding.get("implementation_path") == GPU_LOCK_IMPLEMENTATION_PATH
        and binding.get("nonblocking") is True
        and binding.get("persistent_inode") is True
        and type(binding.get("device")) is int
        and cast(int, binding["device"]) >= 0
        and type(binding.get("inode")) is int
        and cast(int, binding["inode"]) > 0,
        "GPU worker-lease contract drifted.",
    )
    selected_class = binding.get("selected_device_class")
    routing_identity = binding.get("selected_device_routing_identity")
    raw_guard = binding.get("device_guard")
    _require(
        isinstance(selected_class, Mapping)
        and set(selected_class) == execution_environment.SELECTED_DEVICE_CLASS_FIELDS
        and isinstance(selected_class.get("name"), str)
        and bool(cast(str, selected_class["name"]).strip())
        and isinstance(selected_class.get("compute_capability"), list)
        and len(cast(list[Any], selected_class["compute_capability"])) == 2
        and all(
            type(item) is int and item >= 0
            for item in cast(list[Any], selected_class["compute_capability"])
        )
        and type(selected_class.get("total_memory_bytes")) is int
        and cast(int, selected_class["total_memory_bytes"]) > 0
        and isinstance(routing_identity, Mapping)
        and type(binding.get("selected_device_logical_index")) is int
        and cast(int, binding["selected_device_logical_index"]) >= 0
        and binding.get("device_argument") == f"cuda:{binding.get('selected_device_logical_index')}"
        and isinstance(raw_guard, Mapping)
        and set(raw_guard) == _DEVICE_GUARD_BINDING_FIELDS,
        "GPU selected-device or physical guard binding drifted.",
    )
    identity = dict(cast(Mapping[str, Any], routing_identity))
    try:
        expected_guard_path = canonical_device_guard_path(identity)
    except RuntimeError as error:
        raise ValueError("GPU selected-device routing identity drifted.") from error
    guard = cast(Mapping[str, Any], raw_guard)
    raw_guard_path = guard.get("path")
    _require(
        isinstance(raw_guard_path, str)
        and Path(raw_guard_path) == expected_guard_path
        and guard.get("semantics") == DEVICE_GUARD_SEMANTICS
        and guard.get("implementation_path") == GPU_LOCK_IMPLEMENTATION_PATH
        and guard.get("path_derivation") == DEVICE_GUARD_PATH_DERIVATION
        and guard.get("nonblocking") is True
        and guard.get("persistent_inode") is True
        and type(guard.get("device")) is int
        and cast(int, guard["device"]) >= 0
        and type(guard.get("inode")) is int
        and cast(int, guard["inode"]) > 0,
        "GPU physical-device guard contract drifted.",
    )
    if require_live:
        for live_path, device, inode, label in (
            (path, binding["device"], binding["inode"], "scheduler lease"),
            (expected_guard_path, guard["device"], guard["inode"], "physical-device guard"),
        ):
            try:
                metadata = os.stat(live_path, follow_symlinks=False)
            except OSError as error:
                raise ValueError(f"GPU {label} path is inaccessible.") from error
            _require(
                stat.S_ISREG(metadata.st_mode)
                and metadata.st_uid == os.getuid()
                and metadata.st_nlink == 1
                and stat.S_IMODE(metadata.st_mode) == GPU_LOCK_MODE,
                f"GPU {label} path metadata drifted.",
            )
            _require(
                (metadata.st_dev, metadata.st_ino) == (cast(int, device), cast(int, inode)),
                f"GPU {label} persistent identity drifted.",
            )
    return {
        **dict(binding),
        "selected_device_class": dict(cast(Mapping[str, Any], selected_class)),
        "selected_device_routing_identity": identity,
        "device_guard": dict(guard),
    }


def _gpu_device_identity_registry(
    gpu_bindings: Mapping[int, Mapping[str, Any]], *, worker_count: int
) -> list[dict[str, Any]]:
    validated = {
        index: _validate_gpu_lease_binding(binding) for index, binding in gpu_bindings.items()
    }
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    scheduler_to_identity: dict[tuple[str, int, int], tuple[str, str]] = {}
    for worker_index in sorted(validated):
        _require(
            type(worker_index) is int and 0 <= worker_index < worker_count,
            "GPU identity registry contains an invalid worker index.",
        )
        binding = validated[worker_index]
        identity = cast(Mapping[str, Any], binding["selected_device_routing_identity"])
        key = (cast(str, identity["identity_type"]), cast(str, identity["identity"]))
        scheduler = {
            "path": binding["path"],
            "device": binding["device"],
            "inode": binding["inode"],
        }
        scheduler_key = (
            cast(str, scheduler["path"]),
            cast(int, scheduler["device"]),
            cast(int, scheduler["inode"]),
        )
        prior_identity = scheduler_to_identity.setdefault(scheduler_key, key)
        _require(
            prior_identity == key,
            "One scheduler lease identity is assigned to different physical GPUs.",
        )
        candidate = {
            "routing_identity": dict(identity),
            "selected_device_class": dict(
                cast(Mapping[str, Any], binding["selected_device_class"])
            ),
            "scheduler_lease": scheduler,
            "device_guard": dict(cast(Mapping[str, Any], binding["device_guard"])),
            "worker_indices": [worker_index],
            "worker_device_routes": [
                {
                    "worker_index": worker_index,
                    "logical_index": binding["selected_device_logical_index"],
                    "device_argument": binding["device_argument"],
                }
            ],
        }
        prior = by_identity.get(key)
        if prior is None:
            by_identity[key] = candidate
            continue
        _require(
            prior["selected_device_class"] == candidate["selected_device_class"]
            and prior["scheduler_lease"] == scheduler
            and prior["device_guard"] == candidate["device_guard"],
            "One physical GPU routing identity maps to different scheduler or device leases.",
        )
        cast(list[int], prior["worker_indices"]).append(worker_index)
        cast(list[dict[str, Any]], prior["worker_device_routes"]).extend(
            cast(list[dict[str, Any]], candidate["worker_device_routes"])
        )
    return [by_identity[key] for key in sorted(by_identity)]


def _gpu_registry_payload(
    gpu_bindings: Mapping[int, Mapping[str, Any]], *, worker_count: int
) -> list[dict[str, Any]]:
    _require(
        all(type(index) is int and 0 <= index < worker_count for index in gpu_bindings),
        "GPU worker-lease registry contains an invalid worker index.",
    )
    result = [
        {"worker_index": index, **_validate_gpu_lease_binding(gpu_bindings[index])}
        for index in sorted(gpu_bindings)
    ]
    _gpu_device_identity_registry(gpu_bindings, worker_count=worker_count)
    return result


def _distributed_summary_from_ledgers(
    ledgers: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    output_root: Path,
    gpu_bindings: Mapping[int, Mapping[str, Any]],
    worker_count: int,
    prerequisites: FrozenPrerequisites,
    evaluator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    prefix, merged = _merge_worker_records(ledgers, worker_count=worker_count)
    coordinate_items = coordinates()
    storage_records = [merged[index] for index in sorted(merged)]
    storage_remaining = [
        coordinate for index, coordinate in enumerate(coordinate_items) if index not in merged
    ]
    completed_by_worker = {index: len(records) for index, records in ledgers.items()}
    worker_registry = _worker_ledger_registry(
        output_root,
        worker_count=worker_count,
        completed_by_worker=completed_by_worker,
    )
    return (
        _matrix_payload(
            prefix,
            prerequisites=prerequisites,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=matrix_lock_binding,
            worker_count=worker_count,
            gpu_worker_leases=gpu_bindings,
            worker_ledger_root=_worker_ledger_root_binding(output_root, worker_count=worker_count),
            worker_ledger_bindings=worker_registry,
            globally_committed_shards=len(merged),
            storage_records=storage_records,
            storage_remaining_coordinates=storage_remaining,
        ),
        merged,
    )


def _validate_distributed_disk_summary(
    *,
    layout: MatrixLayout,
    expected_payload: Mapping[str, Any],
    worker_count: int,
    prerequisites: FrozenPrerequisites,
    evaluator_script: Path,
    evaluator_binding: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
    expected_gpu_worker_leases: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    _require(layout.matrix_summary.exists(), "Distributed matrix summary disappeared.")
    disk_payload = _load_json_nofollow(layout.matrix_summary, label="controller matrix ledger")
    disk_records = validate_matrix_summary(
        disk_payload,
        output_root=layout.output_root,
        prerequisites=prerequisites,
        evaluator_script=evaluator_script,
        evaluator_binding=evaluator_binding,
        matrix_lock_binding=matrix_lock_binding,
        verify_bundles=False,
        expected_worker_count=worker_count,
        expected_gpu_worker_leases=expected_gpu_worker_leases,
    )
    if disk_payload.get("status") in {"paused_infrastructure", "draining_infrastructure"}:
        reason_field = (
            "infrastructure_pause"
            if disk_payload.get("status") == "paused_infrastructure"
            else "infrastructure_drain"
        )
        pause = disk_payload.get(reason_field)
        _require(
            disk_records == expected_payload["records"]
            and isinstance(pause, Mapping)
            and cast(Mapping[str, Any], pause).get("resume_matrix_payload_sha256")
            == expected_payload.get("payload_sha256"),
            "Blocked distributed matrix does not bind the exact resumable sparse projection.",
        )
    else:
        _require(
            disk_records == expected_payload["records"] and disk_payload == expected_payload,
            "Distributed matrix summary is not the exact sparse-commit/canonical-prefix projection.",
        )
    return disk_payload


def _run_distributed_matrix(
    *,
    layout: MatrixLayout,
    manifest_path: Path,
    evaluator_script: Path,
    attestation_key_path: Path | None,
    max_new_cells: int | None,
    worker_index: int,
    worker_count: int,
    coordinator_only: bool,
    gpu_lock_path: Path,
    _prepared: tuple[
        Path,
        Mapping[str, Any],
        FrozenPrerequisites,
        CanonicalEvaluatorSnapshot,
        Mapping[str, Any],
        GPULockLease | None,
        GPULockLease | None,
        Mapping[str, Any] | None,
    ]
    | None = None,
) -> dict[str, Any]:
    _validate_worker_identity(worker_index=worker_index, worker_count=worker_count)
    _distributed_layout_guard(layout, attestation_key_path=attestation_key_path)
    if _prepared is None:
        canonical = _canonical_evaluator(evaluator_script)
        lock_binding = _initialize_matrix_lock_binding(layout.lock_path)
        gpu_lease = (
            None
            if coordinator_only
            else acquire_gpu_lock(
                f"p2-direct-controller-worker-{worker_index}-of-{worker_count}",
                path=gpu_lock_path,
            )
        )
        device_guard_lease: GPULockLease | None = None
        device_context: dict[str, Any] | None = None
        try:
            if gpu_lease is not None:
                gpu_lease.assert_held()
                device_context = _capture_selected_device_context(gpu_lease)
                device_guard_lease = _acquire_selected_device_guard(
                    label=f"p2-direct-controller-worker-{worker_index}-of-{worker_count}",
                    device_context=device_context,
                    scheduler_lease=gpu_lease,
                )
                device_guard_lease.assert_held()
            prerequisites = load_and_validate_prerequisites(
                manifest_path=manifest_path,
                training_output_root=layout.training_output_root,
                calibration_output_root=layout.calibration_output_root,
                top_p_output_root=layout.top_p_output_root,
                output_root=layout.output_root,
                attestation_key_path=attestation_key_path,
                gpu_lease=gpu_lease,
                device_guard_lease=device_guard_lease,
            )
            if gpu_lease is not None:
                gpu_lease.assert_held()
                _require(
                    device_guard_lease is not None and device_context is not None,
                    "Distributed controller physical-device guard is missing.",
                )
                cast(GPULockLease, device_guard_lease).assert_held()
                device_context = _validate_selected_device_context(
                    cast(Mapping[str, Any], device_context), prerequisites=prerequisites
                )
            _require(
                EVALUATOR_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS
                and GPU_LOCK_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS,
                "Canonical evaluator or GPU lease helper is absent from the frozen inventory.",
            )
            _require(
                contract.implementation_tree_digest()
                == prerequisites.context.manifest_binding["implementation_digest"],
                "Canonical evaluator is not bound by the frozen manifest.",
            )
            evaluator_fd, evaluator_snapshot = _open_evaluator(canonical)
            os.close(evaluator_fd)
            evaluator_binding = evaluator_snapshot.public_binding
            gpu_binding = (
                None
                if gpu_lease is None
                else _gpu_lease_binding(
                    gpu_lease,
                    device_guard_lease=cast(GPULockLease, device_guard_lease),
                    device_context=cast(Mapping[str, Any], device_context),
                )
            )
            prepared_result = _run_distributed_matrix(
                layout=layout,
                manifest_path=manifest_path,
                evaluator_script=evaluator_script,
                attestation_key_path=attestation_key_path,
                max_new_cells=max_new_cells,
                worker_index=worker_index,
                worker_count=worker_count,
                coordinator_only=coordinator_only,
                gpu_lock_path=gpu_lock_path,
                _prepared=(
                    canonical,
                    lock_binding,
                    prerequisites,
                    evaluator_snapshot,
                    evaluator_binding,
                    gpu_lease,
                    device_guard_lease,
                    gpu_binding,
                ),
            )
            if gpu_lease is not None:
                gpu_lease.assert_held()
                cast(GPULockLease, device_guard_lease).assert_held()
            return prepared_result
        finally:
            if gpu_lease is not None:
                if device_guard_lease is None:
                    gpu_lease.close()
                else:
                    _close_gpu_lease_pair(gpu_lease, device_guard_lease)
    canonical = _prepared[0]
    lock_binding = dict(_prepared[1])
    prerequisites = _prepared[2]
    evaluator_snapshot = _prepared[3]
    evaluator_binding = dict(_prepared[4])
    current_gpu_lease = _prepared[5]
    current_device_guard_lease = _prepared[6]
    current_gpu_binding = _prepared[7]
    _require(
        coordinator_only
        == (
            current_gpu_lease is None
            and current_device_guard_lease is None
            and current_gpu_binding is None
        ),
        "Distributed controller GPU lease preparation mode drifted.",
    )
    if current_gpu_lease is not None:
        current_gpu_lease.assert_held()
        cast(GPULockLease, current_device_guard_lease).assert_held()

    with _exclusive_matrix_lock(
        layout.lock_path,
        matrix_summary=layout.matrix_summary,
        expected_binding=lock_binding,
    ):
        ledgers, gpu_bindings = _load_worker_ledgers(
            output_root=layout.output_root,
            worker_count=worker_count,
            prerequisites=prerequisites,
            evaluator_script=canonical,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=lock_binding,
            verify_bundles=coordinator_only,
            expected_worker_index=None if coordinator_only else worker_index,
            expected_gpu_lease_binding=current_gpu_binding,
        )
        # A resuming worker replays its own immutable partition once. Other partitions are
        # accepted through their HMAC ledgers and are independently replayed by their owners.
        if not coordinator_only and worker_index in ledgers:
            own_path = _worker_ledger_path(
                layout.output_root,
                worker_index=worker_index,
                worker_count=worker_count,
            )
            own_records, own_gpu_binding = _validate_worker_ledger(
                _load_json_nofollow(own_path, label="controller worker ledger"),
                worker_index=worker_index,
                worker_count=worker_count,
                output_root=layout.output_root,
                prerequisites=prerequisites,
                evaluator_script=canonical,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
                verify_bundles=True,
                expected_gpu_lease_binding=current_gpu_binding,
            )
            ledgers[worker_index] = own_records
            gpu_bindings[worker_index] = own_gpu_binding
        summary_payload, merged = _distributed_summary_from_ledgers(
            ledgers,
            output_root=layout.output_root,
            gpu_bindings=gpu_bindings,
            worker_count=worker_count,
            prerequisites=prerequisites,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=lock_binding,
        )
        active_claims = _preflight_distributed_output_tree(
            output_root=layout.output_root,
            matrix_summary=layout.matrix_summary,
            merged_records=merged,
            worker_count=worker_count,
        )
        if layout.matrix_summary.exists():
            disk_summary_payload = _validate_distributed_disk_summary(
                layout=layout,
                expected_payload=summary_payload,
                worker_count=worker_count,
                prerequisites=prerequisites,
                evaluator_script=canonical,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
                expected_gpu_worker_leases=gpu_bindings,
            )
            if disk_summary_payload.get("status") in {
                "paused_infrastructure",
                "draining_infrastructure",
            }:
                reason_field = (
                    "infrastructure_pause"
                    if disk_summary_payload.get("status") == "paused_infrastructure"
                    else "infrastructure_drain"
                )
                pause = cast(Mapping[str, Any], disk_summary_payload[reason_field])
                if coordinator_only or pause.get("pause_worker_index") != worker_index:
                    return disk_summary_payload
        else:
            _atomic_write_json(layout.matrix_summary, summary_payload)
        if coordinator_only:
            _require(
                not active_claims,
                "Coordinator-only finalization refuses live distributed cell claims.",
            )
            _require(
                summary_payload.get("status") == "terminal"
                and len(ledgers) == worker_count
                and all(
                    len(ledgers[index])
                    == len(_assigned_coordinates(worker_index=index, worker_count=worker_count))
                    for index in range(worker_count)
                ),
                "Coordinator-only finalization requires every worker ledger to be terminal.",
            )
            _atomic_write_json(layout.matrix_summary, summary_payload)
            validate_matrix_summary(
                summary_payload,
                output_root=layout.output_root,
                prerequisites=prerequisites,
                evaluator_script=canonical,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
                verify_bundles=False,
                expected_worker_count=worker_count,
                expected_gpu_worker_leases=gpu_bindings,
            )
            return summary_payload
        _require(
            not any(index % worker_count == worker_index for index in active_claims),
            "This distributed worker already owns a live cell claim.",
        )
        if worker_index not in ledgers:
            if current_gpu_binding is None:
                raise ValueError("Distributed worker cannot initialize without a held GPU lease.")
            ledgers[worker_index] = []
            gpu_bindings[worker_index] = dict(current_gpu_binding)
            _atomic_write_json(
                _worker_ledger_path(
                    layout.output_root,
                    worker_index=worker_index,
                    worker_count=worker_count,
                ),
                _worker_payload(
                    [],
                    worker_index=worker_index,
                    worker_count=worker_count,
                    prerequisites=prerequisites,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                    gpu_lease_binding=current_gpu_binding,
                ),
            )
            summary_payload, _ = _distributed_summary_from_ledgers(
                ledgers,
                output_root=layout.output_root,
                gpu_bindings=gpu_bindings,
                worker_count=worker_count,
                prerequisites=prerequisites,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
            )
            _atomic_write_json(layout.matrix_summary, summary_payload)

    assigned = _assigned_coordinates(worker_index=worker_index, worker_count=worker_count)
    new_cells = 0
    while True:
        if max_new_cells is not None and new_cells >= max_new_cells:
            break
        claim_manager: Any | None = None
        claim_binding: Mapping[str, Any] | None = None
        claim_active = False
        coordinate: ShardCoordinate | None = None
        command: list[str] | None = None
        launch_nonce: str | None = None
        projected_shards = 0
        projected_token_rows = 0
        blocked_after_commit = False
        with _exclusive_matrix_lock(
            layout.lock_path,
            matrix_summary=layout.matrix_summary,
            expected_binding=lock_binding,
        ):
            ledgers, gpu_bindings = _load_worker_ledgers(
                output_root=layout.output_root,
                worker_count=worker_count,
                prerequisites=prerequisites,
                evaluator_script=canonical,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
                verify_bundles=False,
                expected_worker_index=worker_index,
                expected_gpu_lease_binding=current_gpu_binding,
            )
            current_records = ledgers.get(worker_index)
            if current_records is None:
                raise ValueError("Distributed worker ledger disappeared.")
            summary_payload, merged = _distributed_summary_from_ledgers(
                ledgers,
                output_root=layout.output_root,
                gpu_bindings=gpu_bindings,
                worker_count=worker_count,
                prerequisites=prerequisites,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
            )
            disk_summary_payload = _validate_distributed_disk_summary(
                layout=layout,
                expected_payload=summary_payload,
                worker_count=worker_count,
                prerequisites=prerequisites,
                evaluator_script=canonical,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
                expected_gpu_worker_leases=gpu_bindings,
            )
            active_claims = _preflight_distributed_output_tree(
                output_root=layout.output_root,
                matrix_summary=layout.matrix_summary,
                merged_records=merged,
                worker_count=worker_count,
            )
            _require(
                not any(index % worker_count == worker_index for index in active_claims),
                "This distributed worker already owns a live cell claim.",
            )
            if len(current_records) == len(assigned):
                break
            coordinate = assigned[len(current_records)]
            coordinate_index = coordinates().index(coordinate)
            _require(
                coordinate_index not in merged,
                "Distributed worker's next coordinate is already committed elsewhere.",
            )
            _headroom_preflight_with_attested_pause(
                output_root=layout.output_root,
                matrix_summary=layout.matrix_summary,
                storage_records=[merged[index] for index in sorted(merged)],
                coordinate=coordinate,
                current_payload=disk_summary_payload,
                resume_payload=summary_payload,
                worker_index=worker_index,
                active_claim_coordinate_indices=tuple(sorted(active_claims)),
                trust_root=prerequisites.trust_root,
            )
            _assert_empty_bundle(layout.output_root, coordinate)
            incomplete = [item for index, item in enumerate(coordinates()) if index not in merged]
            projected_shards = len(incomplete)
            projected_token_rows = projected_decode_token_rows(incomplete)
            inputs = prerequisites.bundles[
                (coordinate.scale, coordinate.training_seed, coordinate.budget)
            ]
            launch_nonce = secrets.token_hex(32)
            command = build_evaluator_command(
                evaluator_script=canonical,
                envelope_path=shard_envelope_path(layout.output_root, coordinate),
                manifest_path=prerequisites.context.manifest_path,
                coordinate=coordinate,
                inputs=inputs,
                launch_nonce=launch_nonce,
                gpu_lease_binding=cast(Mapping[str, Any], current_gpu_binding),
            )
            claim_manager = _exclusive_cell_claim(
                shard_output_dir(layout.output_root, coordinate),
                coordinate=coordinate,
                launch_nonce=launch_nonce,
                worker_index=worker_index,
                worker_count=worker_count,
            )
            claim_binding = cast(Mapping[str, Any], claim_manager.__enter__())
            _assert_cell_claim_binding(claim_binding)
            claim_active = True

        _require(
            coordinate is not None and command is not None and launch_nonce is not None,
            "Distributed claim preparation did not produce a launch.",
        )
        try:
            _require(claim_binding is not None, "Distributed cell claim binding is missing.")
            _assert_cell_claim_binding(claim_binding)
            training_matrix.assert_environment_unchanged(prerequisites.context)
            if current_gpu_lease is None:
                raise ValueError("Distributed evaluator launch lacks its worker GPU lease.")
            active_gpu_lease = current_gpu_lease
            active_gpu_lease.assert_held()
            cast(GPULockLease, current_device_guard_lease).assert_held()
            result = _run_evaluator_from_snapshot(
                command,
                canonical=canonical,
                expected=evaluator_snapshot,
                trust_root=prerequisites.trust_root,
                gpu_lease=active_gpu_lease,
                device_guard_lease=cast(GPULockLease, current_device_guard_lease),
                projected_shards=projected_shards,
                projected_token_rows=projected_token_rows,
            )
            active_gpu_lease.assert_held()
            cast(GPULockLease, current_device_guard_lease).assert_held()
            verification_fd, _ = _open_evaluator(canonical, expected=evaluator_snapshot)
            os.close(verification_fd)
            training_matrix.assert_environment_unchanged(prerequisites.context)
            with _exclusive_matrix_lock(
                layout.lock_path,
                matrix_summary=layout.matrix_summary,
                expected_binding=lock_binding,
            ):
                ledgers, gpu_bindings = _load_worker_ledgers(
                    output_root=layout.output_root,
                    worker_count=worker_count,
                    prerequisites=prerequisites,
                    evaluator_script=canonical,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                    verify_bundles=False,
                    expected_worker_index=worker_index,
                    expected_gpu_lease_binding=current_gpu_binding,
                )
                commit_records = ledgers.get(worker_index)
                if commit_records is None:
                    raise ValueError("Distributed worker ledger disappeared.")
                _summary_before, merged_before = _distributed_summary_from_ledgers(
                    ledgers,
                    output_root=layout.output_root,
                    gpu_bindings=gpu_bindings,
                    worker_count=worker_count,
                    prerequisites=prerequisites,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                )
                disk_summary_before = _validate_distributed_disk_summary(
                    layout=layout,
                    expected_payload=_summary_before,
                    worker_count=worker_count,
                    prerequisites=prerequisites,
                    evaluator_script=canonical,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                    expected_gpu_worker_leases=gpu_bindings,
                )
                active_claims = _preflight_distributed_output_tree(
                    output_root=layout.output_root,
                    matrix_summary=layout.matrix_summary,
                    merged_records=merged_before,
                    worker_count=worker_count,
                )
                coordinate_index = coordinates().index(coordinate)
                claim = active_claims.get(coordinate_index)
                _require(
                    claim is not None and claim.get("launch_nonce") == launch_nonce,
                    "Distributed cell claim changed before commit.",
                )
                envelope_path = shard_envelope_path(layout.output_root, coordinate)
                if not envelope_path.is_file():
                    if result.returncode not in {0, 2}:
                        raise subprocess.CalledProcessError(result.returncode, command)
                    raise ValueError(
                        f"Evaluator did not publish its terminal envelope: {coordinate.key}."
                    )
                inputs = prerequisites.bundles[
                    (coordinate.scale, coordinate.training_seed, coordinate.budget)
                ]
                payload = load_and_validate_shard_bundle(
                    envelope_path,
                    coordinate=coordinate,
                    inputs=inputs,
                    launch_nonce=launch_nonce,
                    trust_root=prerequisites.trust_root,
                    gpu_lease_binding=cast(Mapping[str, Any], current_gpu_binding),
                    execution_environment_projection=cast(
                        Mapping[str, Any],
                        prerequisites.public_binding["execution_environment_projection"],
                    ),
                )
                _require(
                    result.returncode
                    == _expected_exit_code(cast(str, payload["terminal_decision"])),
                    "Evaluator exit code does not match its attested integrity decision.",
                )
                record = _run_record(
                    payload,
                    coordinate=coordinate,
                    envelope_path=envelope_path,
                    inputs=inputs,
                    launch_nonce=launch_nonce,
                    command=command,
                )
                _require(
                    len(commit_records) < len(assigned)
                    and assigned[len(commit_records)] == coordinate
                    and coordinate_index not in merged_before,
                    "Distributed worker ledger advanced during its cell execution.",
                )
                active_gpu_lease.assert_held()
                cast(GPULockLease, current_device_guard_lease).assert_held()
                _assert_cell_claim_binding(claim_binding)
                claim_active = False
                cast(Any, claim_manager).__exit__(None, None, None)
                _require(
                    not os.path.lexists(Path(cast(str, claim_binding["path"]))),
                    "Distributed cell claim survived validated release before commit.",
                )
                claim_path = Path(cast(str, claim_binding["path"]))
                prior_records = list(commit_records)
                prior_worker_payload = _worker_payload(
                    prior_records,
                    worker_index=worker_index,
                    worker_count=worker_count,
                    prerequisites=prerequisites,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                    gpu_lease_binding=cast(Mapping[str, Any], current_gpu_binding),
                )
                commit_records.append(record)
                worker_path = _worker_ledger_path(
                    layout.output_root,
                    worker_index=worker_index,
                    worker_count=worker_count,
                )
                try:
                    _atomic_write_json(
                        worker_path,
                        _worker_payload(
                            commit_records,
                            worker_index=worker_index,
                            worker_count=worker_count,
                            prerequisites=prerequisites,
                            evaluator_binding=evaluator_binding,
                            matrix_lock_binding=lock_binding,
                            gpu_lease_binding=cast(Mapping[str, Any], current_gpu_binding),
                        ),
                    )
                    _require(
                        not os.path.lexists(claim_path),
                        "Distributed cell claim reappeared during worker-ledger commit.",
                    )
                    ledgers[worker_index] = commit_records
                    gpu_bindings[worker_index] = dict(cast(Mapping[str, Any], current_gpu_binding))
                    summary_payload, _merged_after = _distributed_summary_from_ledgers(
                        ledgers,
                        output_root=layout.output_root,
                        gpu_bindings=gpu_bindings,
                        worker_count=worker_count,
                        prerequisites=prerequisites,
                        evaluator_binding=evaluator_binding,
                        matrix_lock_binding=lock_binding,
                    )
                    if summary_payload["status"] == "terminal":
                        terminal_ledgers, terminal_gpu_bindings = _load_worker_ledgers(
                            output_root=layout.output_root,
                            worker_count=worker_count,
                            prerequisites=prerequisites,
                            evaluator_script=canonical,
                            evaluator_binding=evaluator_binding,
                            matrix_lock_binding=lock_binding,
                            verify_bundles=True,
                            expected_worker_index=worker_index,
                            expected_gpu_lease_binding=current_gpu_binding,
                        )
                        summary_payload, _merged_after = _distributed_summary_from_ledgers(
                            terminal_ledgers,
                            output_root=layout.output_root,
                            gpu_bindings=terminal_gpu_bindings,
                            worker_count=worker_count,
                            prerequisites=prerequisites,
                            evaluator_binding=evaluator_binding,
                            matrix_lock_binding=lock_binding,
                        )
                        _require(
                            summary_payload["status"] == "terminal",
                            "Terminal coordinator replay lost a worker record.",
                        )
                    remaining_active_claims = _preflight_distributed_output_tree(
                        output_root=layout.output_root,
                        matrix_summary=layout.matrix_summary,
                        merged_records=_merged_after,
                        worker_count=worker_count,
                    )
                    published_payload = summary_payload
                    if disk_summary_before.get("status") in {
                        "paused_infrastructure",
                        "draining_infrastructure",
                    }:
                        published_payload = _rebind_infrastructure_after_commit(
                            prior_blocked_payload=disk_summary_before,
                            resume_payload=summary_payload,
                            ledgers=ledgers,
                            merged_records=_merged_after,
                            active_claim_coordinate_indices=tuple(sorted(remaining_active_claims)),
                            output_root=layout.output_root,
                            worker_count=worker_count,
                            trust_root=prerequisites.trust_root,
                        )
                    blocked_after_commit = published_payload.get("status") in {
                        "paused_infrastructure",
                        "draining_infrastructure",
                    }
                    _atomic_write_json(layout.matrix_summary, published_payload)
                    _require(
                        not os.path.lexists(claim_path),
                        "Distributed cell claim reappeared during authoritative commit.",
                    )
                    verified_active_claims = _preflight_distributed_output_tree(
                        output_root=layout.output_root,
                        matrix_summary=layout.matrix_summary,
                        merged_records=_merged_after,
                        worker_count=worker_count,
                    )
                    _require(
                        set(verified_active_claims) == set(remaining_active_claims),
                        "Distributed active claims changed during authoritative commit.",
                    )
                    active_gpu_lease.assert_held()
                    cast(GPULockLease, current_device_guard_lease).assert_held()
                except BaseException:
                    _atomic_write_json(worker_path, prior_worker_payload)
                    _atomic_write_json(layout.matrix_summary, disk_summary_before)
                    raise
            new_cells += 1
            if blocked_after_commit:
                break
        except BaseException as error:
            if claim_active:
                with _exclusive_matrix_lock(
                    layout.lock_path,
                    matrix_summary=layout.matrix_summary,
                    expected_binding=lock_binding,
                ):
                    claim_active = False
                    cast(Any, claim_manager).__exit__(type(error), error, error.__traceback__)
            raise

    with _exclusive_matrix_lock(
        layout.lock_path,
        matrix_summary=layout.matrix_summary,
        expected_binding=lock_binding,
    ):
        ledgers, gpu_bindings = _load_worker_ledgers(
            output_root=layout.output_root,
            worker_count=worker_count,
            prerequisites=prerequisites,
            evaluator_script=canonical,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=lock_binding,
            verify_bundles=False,
            expected_worker_index=worker_index,
            expected_gpu_lease_binding=current_gpu_binding,
        )
        terminal_or_prefix, merged = _distributed_summary_from_ledgers(
            ledgers,
            output_root=layout.output_root,
            gpu_bindings=gpu_bindings,
            worker_count=worker_count,
            prerequisites=prerequisites,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=lock_binding,
        )
        disk_terminal_or_prefix = _validate_distributed_disk_summary(
            layout=layout,
            expected_payload=terminal_or_prefix,
            worker_count=worker_count,
            prerequisites=prerequisites,
            evaluator_script=canonical,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=lock_binding,
            expected_gpu_worker_leases=gpu_bindings,
        )
        final_active_claims = _preflight_distributed_output_tree(
            output_root=layout.output_root,
            matrix_summary=layout.matrix_summary,
            merged_records=merged,
            worker_count=worker_count,
        )
        if disk_terminal_or_prefix.get("status") == "draining_infrastructure":
            drain = cast(Mapping[str, Any], disk_terminal_or_prefix["infrastructure_drain"])
            _require(
                drain.get("active_claim_coordinate_indices") == sorted(final_active_claims),
                "Infrastructure drain does not bind the exact live claim inventory.",
            )
        elif disk_terminal_or_prefix.get("status") == "paused_infrastructure":
            _require(
                not final_active_claims,
                "Infrastructure pause may not coexist with live distributed claims.",
            )
        else:
            _atomic_write_json(layout.matrix_summary, terminal_or_prefix)
        validate_matrix_summary(
            disk_terminal_or_prefix,
            output_root=layout.output_root,
            prerequisites=prerequisites,
            evaluator_script=canonical,
            evaluator_binding=evaluator_binding,
            matrix_lock_binding=lock_binding,
            verify_bundles=False,
            expected_worker_count=worker_count,
            expected_gpu_worker_leases=gpu_bindings,
        )
        return disk_terminal_or_prefix


def run_matrix(
    *,
    manifest_path: Path = contract.MANIFEST_PATH,
    training_output_root: Path = TRAINING_OUTPUT_ROOT,
    calibration_output_root: Path = CALIBRATION_OUTPUT_ROOT,
    top_p_output_root: Path = TOP_P_OUTPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    matrix_summary: Path | None = None,
    evaluator_script: Path = EVALUATOR_SCRIPT,
    attestation_key_path: Path | None = None,
    max_new_cells: int | None = None,
    worker_index: int = 0,
    worker_count: int = 1,
    coordinator_only: bool = False,
    gpu_lock_path: Path = DEFAULT_GPU_LOCK_PATH,
) -> dict[str, Any]:
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
        top_p_output_root=top_p_output_root,
        attestation_key_path=key_path_for_layout,
    )
    _require(
        not _paths_overlap(
            _absolute(gpu_lock_path).resolve(strict=False),
            _absolute(calibration_matrix.SUPERSEDED_OUTPUT_ROOT).resolve(strict=False),
        ),
        "Controller GPU lock may not overlap the immutable calibration quarantine.",
    )
    _require(
        type(worker_count) is int and worker_count >= 1,
        "worker-count must be a positive integer.",
    )
    _require(
        type(worker_index) is int and 0 <= worker_index < worker_count,
        "worker-index is outside worker-count.",
    )
    _require(
        not coordinator_only or worker_count > 1,
        "coordinator-only requires distributed worker-count > 1.",
    )
    if worker_count > 1:
        return _run_distributed_matrix(
            layout=layout,
            manifest_path=manifest_path,
            evaluator_script=evaluator_script,
            attestation_key_path=attestation_key_path,
            max_new_cells=max_new_cells,
            worker_index=worker_index,
            worker_count=worker_count,
            coordinator_only=coordinator_only,
            gpu_lock_path=gpu_lock_path,
        )
    _require(worker_index == 0, "Single-worker execution requires worker-index 0.")
    _require(
        not _worker_ledger_root(layout.output_root).exists(),
        "Single-worker execution refuses an existing distributed worker-ledger root.",
    )
    canonical = _canonical_evaluator(evaluator_script)
    lock_binding = _initialize_matrix_lock_binding(layout.lock_path)
    gpu_lease = acquire_gpu_lock("p2-direct-controller-matrix-single", path=gpu_lock_path)
    with ExitStack() as execution_stack:
        execution_stack.callback(gpu_lease.close)
        gpu_lease.assert_held()
        device_context = _capture_selected_device_context(gpu_lease)
        device_guard_lease = _acquire_selected_device_guard(
            label="p2-direct-controller-matrix-single",
            device_context=device_context,
            scheduler_lease=gpu_lease,
        )
        if device_guard_lease is not gpu_lease:
            execution_stack.callback(device_guard_lease.close)
        device_guard_lease.assert_held()
        prerequisites = load_and_validate_prerequisites(
            manifest_path=manifest_path,
            training_output_root=layout.training_output_root,
            calibration_output_root=layout.calibration_output_root,
            top_p_output_root=layout.top_p_output_root,
            output_root=layout.output_root,
            attestation_key_path=attestation_key_path,
            gpu_lease=gpu_lease,
            device_guard_lease=device_guard_lease,
        )
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
        device_context = _validate_selected_device_context(
            device_context, prerequisites=prerequisites
        )
        _require(
            EVALUATOR_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS
            and GPU_LOCK_IMPLEMENTATION_PATH in contract.IMPLEMENTATION_PATHS,
            "Canonical evaluator or GPU lease helper is absent from the frozen inventory.",
        )
        _require(
            contract.implementation_tree_digest()
            == prerequisites.context.manifest_binding["implementation_digest"],
            "Canonical evaluator is not bound by the frozen manifest.",
        )
        evaluator_fd, evaluator_snapshot = _open_evaluator(canonical)
        os.close(evaluator_fd)
        evaluator_binding = evaluator_snapshot.public_binding
        gpu_binding = _gpu_lease_binding(
            gpu_lease,
            device_guard_lease=device_guard_lease,
            device_context=device_context,
        )
        completed: list[dict[str, Any]] = []
        with _exclusive_matrix_lock(
            layout.lock_path,
            matrix_summary=layout.matrix_summary,
            expected_binding=lock_binding,
        ):
            _require(
                not _worker_ledger_root(layout.output_root).exists(),
                "Single-worker execution refuses an existing distributed worker-ledger root.",
            )
            if layout.matrix_summary.exists():
                completed = validate_matrix_summary(
                    _load_json_nofollow(layout.matrix_summary, label="controller matrix ledger"),
                    output_root=layout.output_root,
                    prerequisites=prerequisites,
                    evaluator_script=canonical,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                    verify_bundles=True,
                    expected_gpu_worker_leases={0: gpu_binding},
                )
            _preflight_output_tree(
                output_root=layout.output_root,
                matrix_summary=layout.matrix_summary,
                completed_shards=len(completed),
            )
            training_matrix.assert_environment_unchanged(prerequisites.context)
            verification_fd, _ = _open_evaluator(canonical, expected=evaluator_snapshot)
            os.close(verification_fd)
            if not layout.matrix_summary.exists():
                _atomic_write_json(
                    layout.matrix_summary,
                    _matrix_payload(
                        completed,
                        prerequisites=prerequisites,
                        evaluator_binding=evaluator_binding,
                        matrix_lock_binding=lock_binding,
                        gpu_worker_leases={0: gpu_binding},
                    ),
                )

        new_cells = 0
        while len(completed) < len(coordinates()):
            if max_new_cells is not None and new_cells >= max_new_cells:
                break
            claim_manager: Any | None = None
            claim_binding: Mapping[str, Any] | None = None
            claim_active = False
            coordinate = coordinates()[len(completed)]
            inputs = prerequisites.bundles[
                (coordinate.scale, coordinate.training_seed, coordinate.budget)
            ]
            launch_nonce = secrets.token_hex(32)
            envelope_path = shard_envelope_path(layout.output_root, coordinate)
            command = build_evaluator_command(
                evaluator_script=canonical,
                envelope_path=envelope_path,
                manifest_path=prerequisites.context.manifest_path,
                coordinate=coordinate,
                inputs=inputs,
                launch_nonce=launch_nonce,
                gpu_lease_binding=gpu_binding,
            )
            with _exclusive_matrix_lock(
                layout.lock_path,
                matrix_summary=layout.matrix_summary,
                expected_binding=lock_binding,
            ):
                disk_payload = _load_json_nofollow(
                    layout.matrix_summary, label="controller matrix ledger"
                )
                disk_completed = validate_matrix_summary(
                    disk_payload,
                    output_root=layout.output_root,
                    prerequisites=prerequisites,
                    evaluator_script=canonical,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                    verify_bundles=False,
                    expected_gpu_worker_leases={0: gpu_binding},
                )
                _require(
                    disk_completed == completed,
                    "Single-worker matrix advanced during cell preparation.",
                )
                _preflight_output_tree(
                    output_root=layout.output_root,
                    matrix_summary=layout.matrix_summary,
                    completed_shards=len(completed),
                )
                training_matrix.assert_environment_unchanged(prerequisites.context)
                resume_payload = _matrix_payload(
                    completed,
                    prerequisites=prerequisites,
                    evaluator_binding=evaluator_binding,
                    matrix_lock_binding=lock_binding,
                    gpu_worker_leases={0: gpu_binding},
                )
                _headroom_preflight_with_attested_pause(
                    output_root=layout.output_root,
                    matrix_summary=layout.matrix_summary,
                    storage_records=completed,
                    coordinate=coordinate,
                    current_payload=disk_payload,
                    resume_payload=resume_payload,
                    worker_index=0,
                    trust_root=prerequisites.trust_root,
                )
                _assert_empty_bundle(layout.output_root, coordinate)
                claim_manager = _exclusive_cell_claim(
                    shard_output_dir(layout.output_root, coordinate),
                    coordinate=coordinate,
                    launch_nonce=launch_nonce,
                )
                claim_binding = cast(Mapping[str, Any], claim_manager.__enter__())
                _assert_cell_claim_binding(claim_binding)
                claim_active = True

            try:
                _require(claim_binding is not None, "Single-worker cell claim binding is missing.")
                _assert_cell_claim_binding(claim_binding)
                gpu_lease.assert_held()
                result = _run_evaluator_from_snapshot(
                    command,
                    canonical=canonical,
                    expected=evaluator_snapshot,
                    trust_root=prerequisites.trust_root,
                    gpu_lease=gpu_lease,
                    device_guard_lease=device_guard_lease,
                    projected_shards=EXPECTED_SHARDS - len(completed),
                    projected_token_rows=projected_decode_token_rows(
                        coordinates()[len(completed) :]
                    ),
                )
                gpu_lease.assert_held()
                device_guard_lease.assert_held()
                training_matrix.assert_environment_unchanged(prerequisites.context)
                _assert_cell_claim_binding(claim_binding)
                with _exclusive_matrix_lock(
                    layout.lock_path,
                    matrix_summary=layout.matrix_summary,
                    expected_binding=lock_binding,
                ):
                    disk_completed = validate_matrix_summary(
                        _load_json_nofollow(
                            layout.matrix_summary, label="controller matrix ledger"
                        ),
                        output_root=layout.output_root,
                        prerequisites=prerequisites,
                        evaluator_script=canonical,
                        evaluator_binding=evaluator_binding,
                        matrix_lock_binding=lock_binding,
                        verify_bundles=False,
                        expected_gpu_worker_leases={0: gpu_binding},
                    )
                    _require(
                        disk_completed == completed,
                        "Single-worker matrix advanced during cell execution.",
                    )
                    _assert_cell_claim_binding(claim_binding)
                    if not envelope_path.is_file():
                        if result.returncode not in {0, 2}:
                            raise subprocess.CalledProcessError(result.returncode, command)
                        raise ValueError(
                            f"Evaluator did not publish its terminal envelope: {coordinate.key}."
                        )
                    payload = load_and_validate_shard_bundle(
                        envelope_path,
                        coordinate=coordinate,
                        inputs=inputs,
                        launch_nonce=launch_nonce,
                        trust_root=prerequisites.trust_root,
                        gpu_lease_binding=gpu_binding,
                        execution_environment_projection=cast(
                            Mapping[str, Any],
                            prerequisites.public_binding["execution_environment_projection"],
                        ),
                    )
                    _require(
                        result.returncode
                        == _expected_exit_code(cast(str, payload["terminal_decision"])),
                        "Evaluator exit code does not match its attested integrity decision.",
                    )
                    record = _run_record(
                        payload,
                        coordinate=coordinate,
                        envelope_path=envelope_path,
                        inputs=inputs,
                        launch_nonce=launch_nonce,
                        command=command,
                    )
                    gpu_lease.assert_held()
                    device_guard_lease.assert_held()
                    _assert_cell_claim_binding(claim_binding)
                    claim_active = False
                    cast(Any, claim_manager).__exit__(None, None, None)
                    claim_path = Path(cast(str, claim_binding["path"]))
                    _require(
                        not os.path.lexists(claim_path),
                        "Single-worker cell claim survived validated release before commit.",
                    )
                    promoted = [*completed, record]
                    promoted_payload = _matrix_payload(
                        promoted,
                        prerequisites=prerequisites,
                        evaluator_binding=evaluator_binding,
                        matrix_lock_binding=lock_binding,
                        gpu_worker_leases={0: gpu_binding},
                    )
                    prior_payload = _matrix_payload(
                        completed,
                        prerequisites=prerequisites,
                        evaluator_binding=evaluator_binding,
                        matrix_lock_binding=lock_binding,
                        gpu_worker_leases={0: gpu_binding},
                    )
                    try:
                        _atomic_write_json(layout.matrix_summary, promoted_payload)
                        _require(
                            not os.path.lexists(claim_path),
                            "Single-worker cell claim reappeared during authoritative commit.",
                        )
                        _preflight_output_tree(
                            output_root=layout.output_root,
                            matrix_summary=layout.matrix_summary,
                            completed_shards=len(promoted),
                        )
                        validate_matrix_summary(
                            promoted_payload,
                            output_root=layout.output_root,
                            prerequisites=prerequisites,
                            evaluator_script=canonical,
                            evaluator_binding=evaluator_binding,
                            matrix_lock_binding=lock_binding,
                            verify_bundles=False,
                            expected_gpu_worker_leases={0: gpu_binding},
                        )
                        gpu_lease.assert_held()
                        device_guard_lease.assert_held()
                    except BaseException:
                        _atomic_write_json(layout.matrix_summary, prior_payload)
                        raise
                    completed = promoted
                new_cells += 1
            except BaseException as error:
                if claim_active:
                    with _exclusive_matrix_lock(
                        layout.lock_path,
                        matrix_summary=layout.matrix_summary,
                        expected_binding=lock_binding,
                    ):
                        claim_active = False
                        cast(Any, claim_manager).__exit__(type(error), error, error.__traceback__)
                raise

        with _exclusive_matrix_lock(
            layout.lock_path,
            matrix_summary=layout.matrix_summary,
            expected_binding=lock_binding,
        ):
            terminal_or_prefix = _matrix_payload(
                completed,
                prerequisites=prerequisites,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
                gpu_worker_leases={0: gpu_binding},
            )
            disk_payload = _load_json_nofollow(
                layout.matrix_summary, label="controller matrix ledger"
            )
            _require(
                disk_payload == terminal_or_prefix,
                "Single-worker terminal ledger differs from its authenticated prefix.",
            )
            validate_matrix_summary(
                terminal_or_prefix,
                output_root=layout.output_root,
                prerequisites=prerequisites,
                evaluator_script=canonical,
                evaluator_binding=evaluator_binding,
                matrix_lock_binding=lock_binding,
                verify_bundles=False,
                expected_gpu_worker_leases={0: gpu_binding},
            )
            _preflight_output_tree(
                output_root=layout.output_root,
                matrix_summary=layout.matrix_summary,
                completed_shards=len(completed),
            )
        gpu_lease.assert_held()
        device_guard_lease.assert_held()
        return terminal_or_prefix


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resume-safe frozen 9,000-shard direct-controller quality matrix."
    )
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument("--training-output-root", type=Path, default=TRAINING_OUTPUT_ROOT)
    parser.add_argument("--calibration-output-root", type=Path, default=CALIBRATION_OUTPUT_ROOT)
    parser.add_argument("--top-p-output-root", type=Path, default=TOP_P_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--matrix-summary", type=Path)
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument(
        "--gpu-lock-path",
        type=Path,
        default=DEFAULT_GPU_LOCK_PATH,
        help="Device-scoped nonblocking scheduler lease path; use one distinct path per GPU.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Zero-based local worker index; distributed claims use /proc PID identity.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help=(
            "Worker count on one host with a shared local filesystem; multi-host PID claims "
            "are intentionally unsupported."
        ),
    )
    parser.add_argument(
        "--coordinator-only",
        action="store_true",
        help="Validate and publish a terminal merge only after every local worker ledger completes.",
    )
    args = parser.parse_args()
    try:
        result = run_matrix(
            manifest_path=args.manifest,
            training_output_root=args.training_output_root,
            calibration_output_root=args.calibration_output_root,
            top_p_output_root=args.top_p_output_root,
            output_root=args.output_root,
            matrix_summary=args.matrix_summary,
            evaluator_script=EVALUATOR_SCRIPT,
            max_new_cells=args.max_new_cells,
            worker_index=args.worker_index,
            worker_count=args.worker_count,
            coordinator_only=args.coordinator_only,
            gpu_lock_path=args.gpu_lock_path,
        )
    except InfrastructurePauseSignal as signal:
        print(
            json.dumps(
                {
                    "status": (
                        "draining_infrastructure"
                        if signal.payload.get("drain_rule") == INFRASTRUCTURE_DRAIN_RULE
                        else "paused_infrastructure"
                    ),
                    "retryable": signal.payload.get("retryable"),
                    "sufficient": signal.payload.get("sufficient"),
                    "required_headroom_bytes": signal.payload.get("required_headroom_bytes"),
                    "filesystem_available_bytes": signal.payload.get("filesystem_available_bytes"),
                },
                sort_keys=True,
            )
        )
        return 3
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "integrity_status": result.get("integrity_status"),
                "completed_shards": result["completed_shards"],
                "canonical_prefix_shards": result["canonical_prefix_shards"],
                "globally_committed_shards": result["globally_committed_shards"],
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    if result["status"] in {"paused_infrastructure", "draining_infrastructure"}:
        return 3
    if result["status"] == "terminal" and result["integrity_status"] == "INTEGRITY-FAIL":
        return 2
    return 0


_require(EXPECTED_SHARDS == 9_000, "Frozen direct-controller matrix cardinality drifted.")
_require(EXAMPLES_PER_SHARD == 20, "Frozen direct-controller shard size drifted.")
_require(
    sum(expected_decode_token_rows(item) for item in coordinates())
    == contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES,
    "Frozen direct-controller decode-token projection weight drifted.",
)
_require(len(ARM_NAMES) == 19, "Frozen direct-controller arm inventory drifted.")


if __name__ == "__main__":
    raise SystemExit(main())
