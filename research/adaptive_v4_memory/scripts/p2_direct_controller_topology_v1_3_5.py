from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
import stat
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import p2_direct_controller_contract_v1_3_5 as contract
import p2_direct_controller_reuse_admission_v1_3 as legacy
from adaptive_v4_gpu_lock import GPULockLease, acquire_gpu_lock

QualityContext = legacy.QualityContext
ValidatedReuseAdmission = legacy.ValidatedReuseAdmission
ValidatedPreheldoutGenesis = legacy.ValidatedPreheldoutGenesis
AdmittedCalibration = legacy.AdmittedCalibration
AdmittedCheckpoint = legacy.AdmittedCheckpoint

SAFE_FILE_MODE = 0o600
SAFE_DIRECTORY_MODE = 0o700
V1_3_5_ACTIVATION_SCHEMA_VERSION = 1
V1_3_5_ACTIVATION_MATRIX_LOCK_SEMANTICS = (
    "activation-root-precreated-inode-flock-plus-process-thread-mutex-v2"
)
V1_3_5_ACTIVATION_BOOTSTRAP_LOCK_PATH = Path(
    "/tmp/adaptive-v4-p2-direct-controller-v1-3-5-activation-bootstrap.lock"
)
V1_3_5_ACTIVATION_STAGING_PREFIX = f".{contract.V1_3_5_ACTIVATION_ROOT.name}.staging-"

_GPU_BORROW_GUARD = threading.Lock()
_GPU_BORROW_COUNTS: dict[tuple[int, int, int, int], int] = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _gpu_owner_key(owner: GPULockLease) -> tuple[int, int, int, int]:
    return (id(owner), owner.file_descriptor, owner.device, owner.inode)


class BorrowedGPULockLeaseV1_3_5(GPULockLease):
    """Non-owning thread view over one supervisor-owned persistent lock descriptor."""

    def __init__(self, owner: GPULockLease) -> None:
        super().__init__(
            path=owner.path,
            file_descriptor=owner.file_descriptor,
            device=owner.device,
            inode=owner.inode,
            ctime_ns=owner.ctime_ns,
            mtime_ns=owner.mtime_ns,
            holder=owner.holder,
        )
        self._owner = owner

    def close(self) -> None:
        if self._closed:
            return
        failure: BaseException | None = None
        try:
            self._owner.assert_held()
            self.assert_held()
        except BaseException as error:
            failure = error
        key = _gpu_owner_key(self._owner)
        with _GPU_BORROW_GUARD:
            count = _GPU_BORROW_COUNTS.get(key, 0)
            _require(count > 0, "V1.3.5 GPU borrowed-view accounting underflowed.")
            if count == 1:
                del _GPU_BORROW_COUNTS[key]
            else:
                _GPU_BORROW_COUNTS[key] = count - 1
            self._closed = True
        if failure is not None:
            raise failure


def borrow_gpu_lock_lease(owner: GPULockLease) -> GPULockLease:
    _require(
        type(owner) is GPULockLease and not owner.closed,
        "Only a live owning GPU lease may create a v1.3.5 borrowed view.",
    )
    owner.assert_held()
    key = _gpu_owner_key(owner)
    with _GPU_BORROW_GUARD:
        _GPU_BORROW_COUNTS[key] = _GPU_BORROW_COUNTS.get(key, 0) + 1
    try:
        return BorrowedGPULockLeaseV1_3_5(owner)
    except BaseException:
        with _GPU_BORROW_GUARD:
            count = _GPU_BORROW_COUNTS[key]
            if count == 1:
                del _GPU_BORROW_COUNTS[key]
            else:
                _GPU_BORROW_COUNTS[key] = count - 1
        raise


def active_gpu_lock_borrows(owner: GPULockLease) -> int:
    with _GPU_BORROW_GUARD:
        return _GPU_BORROW_COUNTS.get(_gpu_owner_key(owner), 0)


def close_gpu_lock_owner(owner: GPULockLease) -> None:
    _require(
        active_gpu_lock_borrows(owner) == 0,
        "V1.3.5 GPU owner still has active borrowed views.",
    )
    owner.close()


def _absolute(path: Path, *, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _canonical_payload(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload, opened = legacy._load_json_nofollow(
        path,
        label=label,
        require_canonical_pretty_bytes=True,
    )
    try:
        raw = os.pread(opened.file_descriptor, opened.bytes + 1, 0)
        _require(
            len(raw) == opened.bytes and hashlib.sha256(raw).hexdigest() == opened.sha256,
            f"{label} changed while open.",
        )
        opened.assert_unchanged()
        return dict(payload), raw
    finally:
        opened.close()


def establish_v1_3_5_quality_context(
    manifest_path: Path,
    *,
    implementation_paths: Sequence[str],
    repository_root: Path = legacy.REPOSITORY_ROOT,
) -> QualityContext:
    root = legacy._exact_path(
        repository_root,
        label="Repository root",
        must_exist=True,
    )
    _require(root.is_dir(), "Repository root is not a directory.")
    source = legacy._source_state(root)
    _require(source["dirty"] is False, "Quality context requires a clean checkout.")
    path = legacy._exact_path(
        manifest_path,
        label="Final v1.3.5 manifest",
        repository_root=root,
        must_exist=True,
    )
    payload, raw = _canonical_payload(path, label="Final v1.3.5 manifest")
    validated = contract.validate_v1_3_5_manifest_payload(
        payload,
        verify_implementation=True,
    )
    _require(validated == payload, "Canonical v1.3.5 manifest validator changed payload.")
    implementation = payload.get("implementation")
    public_attestation = payload.get("attestation")
    _require(
        payload.get("experiment_id") == contract.V1_3_5_EXPERIMENT_ID
        and isinstance(implementation, Mapping)
        and isinstance(public_attestation, Mapping),
        "V1.3.5 manifest identity or implementation is invalid.",
    )
    implementation_map = cast(Mapping[str, Any], implementation)
    paths = tuple(implementation_paths)
    source_commit = implementation_map.get("source_commit")
    tree_digest = implementation_map.get("tree_digest")
    _require(
        paths == contract.V1_3_5_IMPLEMENTATION_PATHS
        and tuple(implementation_map.get("paths", ())) == paths
        and contract.base.is_git_oid(source_commit)
        and contract.base.is_sha256(tree_digest),
        "Caller inventory differs from the canonical v1.3.5 contract.",
    )
    current_entries = legacy._index_inventory(root, paths)
    _require(
        legacy._implementation_index_digest(paths, current_entries) == tree_digest,
        "Current v1.3.5 implementation differs from the manifest.",
    )
    frozen_inventory = legacy._commit_inventory(
        root,
        cast(str, source_commit),
        paths,
    )
    _require(
        legacy._commit_inventory_digest(
            frozen_inventory,
            paths,
        )
        == tree_digest,
        "V1.3.5 source commit does not reproduce its tree digest.",
    )
    live = legacy._quality_live_implementation_inventory(
        root,
        paths=paths,
        source_commit=cast(str, source_commit),
        expected_digest=cast(str, tree_digest),
    )
    legacy._assert_ancestor(
        root,
        cast(str, source_commit),
        cast(str, source["commit"]),
    )
    expected_attestation = legacy.attestation.public_manifest_contract(
        str(cast(Mapping[str, Any], public_attestation).get("key_id", ""))
    )
    _require(
        dict(cast(Mapping[str, Any], public_attestation)) == expected_attestation,
        "V1.3.5 manifest attestation contract drifted.",
    )
    binding = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "experiment_id": contract.V1_3_5_EXPERIMENT_ID,
        "implementation_source_commit": source_commit,
        "implementation_digest": tree_digest,
        "live_implementation_inventory_digest": live["digest"],
        "live_implementation_file_count": live["file_count"],
        "attestation": expected_attestation,
    }
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
    legacy.assert_quality_context_unchanged(context)
    _require(
        context.manifest_binding.get("experiment_id") == contract.V1_3_5_EXPERIMENT_ID,
        "Quality context is not v1.3.5.",
    )


def quality_implementation_inventory(
    context: QualityContext,
) -> tuple[dict[str, Any], ...]:
    assert_quality_context_unchanged(context)
    return legacy.quality_implementation_inventory(context)


def validate_admitted_calibration(
    calibration: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    admission: ValidatedReuseAdmission,
    trust_root: legacy.attestation.TrustRoot,
    expected_scale: str,
    expected_training_seed: int,
) -> dict[str, Any]:
    return legacy.validate_admitted_calibration(
        calibration,
        artifact_path=artifact_path,
        admission=admission,
        trust_root=trust_root,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
    )


def load_admitted_checkpoint(
    calibration: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
    admission: ValidatedReuseAdmission,
    trust_root: legacy.attestation.TrustRoot,
    expected_scale: str,
    expected_training_seed: int,
) -> Mapping[str, Any]:
    return legacy.load_admitted_checkpoint(
        calibration,
        artifact_path=artifact_path,
        admission=admission,
        trust_root=trust_root,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
    )


def _static_v1_3_4_bundle(
    *,
    quality_context: QualityContext,
    trust_root: legacy.attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> tuple[ValidatedReuseAdmission, ValidatedPreheldoutGenesis]:
    assert_quality_context_unchanged(quality_context)
    _require(
        trust_root.key_id
        == cast(Mapping[str, Any], quality_context.manifest_binding["attestation"])["key_id"],
        "Static predecessor trust root differs from v1.3.5.",
    )
    admission_path = _absolute(
        contract.V1_3_5_REUSE_ADMISSION_PATH,
        repository_root=quality_context.repository_root,
    )
    genesis_path = _absolute(
        contract.V1_3_5_PREHELDOUT_GENESIS_PATH,
        repository_root=quality_context.repository_root,
    )
    admission_payload, admission_raw = _canonical_payload(
        admission_path,
        label="Static v1.3.4 reuse admission",
    )
    genesis_payload, genesis_raw = _canonical_payload(
        genesis_path,
        label="Static v1.3.4 preheldout genesis",
    )
    _require(
        len(admission_raw) == contract.V1_3_4_STATIC_ADMISSION_BYTES
        and hashlib.sha256(admission_raw).hexdigest() == contract.V1_3_4_STATIC_ADMISSION_SHA256
        and len(genesis_raw) == contract.V1_3_4_STATIC_GENESIS_BYTES
        and hashlib.sha256(genesis_raw).hexdigest() == contract.V1_3_4_STATIC_GENESIS_SHA256,
        "Static v1.3.4 admission/genesis bytes drifted.",
    )
    legacy._verify_attested_payload(
        admission_payload,
        trust_root=trust_root,
        purpose=legacy.V1_3_4_REUSE_ADMISSION_PURPOSE,
        label="Static v1.3.4 reuse admission",
    )
    legacy._verify_attested_payload(
        genesis_payload,
        trust_root=trust_root,
        purpose=legacy.V1_3_4_PREHELDOUT_GENESIS_PURPOSE,
        label="Static v1.3.4 preheldout genesis",
    )
    zero_fields = (
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "active_claim_count",
        "worker_ledger_count",
        "top_p_quality_input_count",
    )
    _require(
        admission_payload.get("experiment_id") == contract.base.V1_3_4_EXPERIMENT_ID
        and genesis_payload.get("experiment_id") == contract.base.V1_3_4_EXPERIMENT_ID
        and admission_payload.get("quality_evaluation_started") is False
        and admission_payload.get("quality_rng_initialized") is False
        and genesis_payload.get("quality_evaluation_started") is False
        and genesis_payload.get("quality_rng_initialized") is False
        and all(admission_payload.get(field) == 0 for field in zero_fields)
        and all(genesis_payload.get(field) == 0 for field in zero_fields)
        and genesis_payload.get("expected_shards") == expected_shards
        and genesis_payload.get("coordinate_digest") == coordinate_digest
        and tuple(genesis_payload.get("exact_fill_arm_names", ())) == tuple(exact_fill_arm_names),
        "Static v1.3.4 predecessor violates the zero-quality/grid boundary.",
    )
    lineage = legacy.load_superseded_empty_lineage_v1_3(
        trust_root=trust_root,
        repository_root=quality_context.repository_root,
    )
    calibrations, checkpoints = legacy._v1_3_4_admission_entries(
        lineage,
        quality_context=quality_context,
    )
    for calibration in calibrations.values():
        legacy._validate_file_binding(
            calibration.public_binding,
            label="Static v1.3.4 admitted calibration",
            repository_root=quality_context.repository_root,
        )
    for checkpoint in checkpoints.values():
        legacy._validate_file_binding(
            checkpoint.public_binding,
            label="Static v1.3.4 admitted checkpoint",
            repository_root=quality_context.repository_root,
        )
    admission_binding = legacy._v1_3_4_reuse_admission_public_binding(
        path=admission_path,
        payload=admission_payload,
        sha256=contract.V1_3_4_STATIC_ADMISSION_SHA256,
        byte_count=contract.V1_3_4_STATIC_ADMISSION_BYTES,
    )
    admission = ValidatedReuseAdmission(
        payload=admission_payload,
        public_binding=admission_binding,
        calibrations=calibrations,
        checkpoints=checkpoints,
        quality_context=quality_context,
        execution_environment_projection=dict(
            cast(Mapping[str, Any], admission_payload["execution_environment_projection"])
        ),
    )
    genesis_binding = legacy._v1_3_4_genesis_public_binding(
        path=genesis_path,
        payload=genesis_payload,
        sha256=contract.V1_3_4_STATIC_GENESIS_SHA256,
        byte_count=contract.V1_3_4_STATIC_GENESIS_BYTES,
    )
    _require(
        genesis_payload.get("reuse_admission") == admission.public_binding,
        "Static v1.3.4 genesis does not bind the exact admission.",
    )
    return admission, ValidatedPreheldoutGenesis(
        payload=genesis_payload,
        public_binding=genesis_binding,
    )


_PRESTART_SEAL = object()
_ACTIVATION_SEAL = object()
_LEASE_SEAL = object()


@dataclass(frozen=True)
class PrestartQualityAuthorityV1_3_5:
    _seal: object
    quality_context: QualityContext
    reuse_admission: ValidatedReuseAdmission
    preheldout_genesis: ValidatedPreheldoutGenesis
    absence_witness: dict[str, Any]


@dataclass(frozen=True)
class ValidatedQualityStartActivationV1_3_5:
    _seal: object
    payload: dict[str, Any]
    public_binding: dict[str, Any]
    quality_context: QualityContext
    reuse_admission: ValidatedReuseAdmission
    preheldout_genesis: ValidatedPreheldoutGenesis
    consumer_coordinate: tuple[str, int] | None
    root_identity: dict[str, Any]
    matrix_lock_binding: dict[str, Any]


@dataclass(frozen=True)
class ActivatedConsumerAuthorityV1_3_5:
    _seal: object
    activation: ValidatedQualityStartActivationV1_3_5
    reuse_admission: ValidatedReuseAdmission
    preheldout_genesis: ValidatedPreheldoutGenesis
    coordinate: tuple[str, int]


_CONSUMER_SEAL = object()


def _prospective_paths(context: QualityContext) -> tuple[Path, ...]:
    return tuple(
        _absolute(path, repository_root=context.repository_root)
        for path in (
            contract.V1_3_5_OUTPUT_ROOT,
            contract.V1_3_5_INTEGRITY_OUTPUT_PATH,
            contract.V1_3_5_SUMMARY_OUTPUT_PATH,
            contract.V1_3_5_WORKER_LEDGER_ROOT,
            contract.V1_3_5_PERSISTENT_SESSION_LEDGER_ROOT,
            contract.V1_3_5_PERSISTENT_SESSION_LEDGER_LOCK_PATH,
            contract.V1_3_5_ACTIVATION_ROOT,
        )
    )


def _absence_witness(context: QualityContext) -> dict[str, Any]:
    absent = [str(path) for path in _prospective_paths(context)]
    _require(
        all(not os.path.lexists(path) for path in absent),
        "A v1.3.5 prospective quality path already exists.",
    )
    source = {
        "schema_version": 1,
        "authority_type": "v1.3.5-live-prestart-absence",
        "quality_source": context.source,
        "quality_manifest": context.manifest_binding,
        "absent_paths": absent,
        "quality_values_accessed": False,
    }
    return {**source, "witness_sha256": contract.base.json_digest(source)}


def _check_absence_witness(context: QualityContext, witness: Mapping[str, Any]) -> None:
    _require(
        dict(witness) == _absence_witness(context),
        "V1.3.5 prestart absence witness changed.",
    )


def load_prestart_quality_authority(
    *,
    quality_context: QualityContext,
    trust_root: legacy.attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> PrestartQualityAuthorityV1_3_5:
    reuse, genesis = _static_v1_3_4_bundle(
        quality_context=quality_context,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    return PrestartQualityAuthorityV1_3_5(
        _seal=_PRESTART_SEAL,
        quality_context=quality_context,
        reuse_admission=reuse,
        preheldout_genesis=genesis,
        absence_witness=_absence_witness(quality_context),
    )


def _require_prestart(
    authority: PrestartQualityAuthorityV1_3_5,
) -> PrestartQualityAuthorityV1_3_5:
    _require(
        type(authority) is PrestartQualityAuthorityV1_3_5
        and authority._seal is _PRESTART_SEAL
        and type(authority.reuse_admission) is ValidatedReuseAdmission
        and type(authority.preheldout_genesis) is ValidatedPreheldoutGenesis,
        "V1.3.5 prestart authority is raw or invalid.",
    )
    assert_quality_context_unchanged(authority.quality_context)
    _check_absence_witness(authority.quality_context, authority.absence_witness)
    return authority


def _activation_root_members(
    storage_root: Path,
    *,
    canonical_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = legacy._exact_path(
        storage_root,
        label="V1.3.5 activation root",
        must_exist=True,
    )
    root_stat = os.stat(root, follow_symlinks=False)
    _require(
        stat.S_ISDIR(root_stat.st_mode)
        and root_stat.st_uid == os.getuid()
        and stat.S_IMODE(root_stat.st_mode) == SAFE_DIRECTORY_MODE
        and set(os.listdir(root))
        == {
            contract.V1_3_5_ACTIVATION_MATRIX_LOCK_PATH.name,
            contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name,
        },
        "V1.3.5 activation root is not safe exact-two.",
    )
    identities: dict[str, os.stat_result] = {}
    for name in sorted(os.listdir(root)):
        path = root / name
        metadata = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE,
            f"Unsafe v1.3.5 activation member: {name}",
        )
        identities[name] = metadata
    lock = identities[contract.V1_3_5_ACTIVATION_MATRIX_LOCK_PATH.name]
    return (
        {
            "path": str(canonical_root),
            "device": root_stat.st_dev,
            "inode": root_stat.st_ino,
            "uid": root_stat.st_uid,
            "gid": root_stat.st_gid,
            "mode": SAFE_DIRECTORY_MODE,
            "nlink": root_stat.st_nlink,
            "persistent_inode": True,
        },
        {
            "path": str(canonical_root / contract.V1_3_5_ACTIVATION_MATRIX_LOCK_PATH.name),
            "semantics": V1_3_5_ACTIVATION_MATRIX_LOCK_SEMANTICS,
            "persistent_inode": True,
            "device": lock.st_dev,
            "inode": lock.st_ino,
            "uid": lock.st_uid,
            "mode": SAFE_FILE_MODE,
            "nlink": lock.st_nlink,
            "unlink_on_release": False,
        },
        root / contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name,
    )


def _manifest_topology(context: QualityContext) -> tuple[int, dict[str, Any]]:
    payload = contract.load_v1_3_5_manifest(
        context.manifest_path,
        verify_implementation=True,
    )
    execution = cast(Mapping[str, Any], payload["execution_contract"])
    sealed = cast(Mapping[str, Any], execution["sealed_launch_and_persistent_session"])
    activation = cast(Mapping[str, Any], sealed["quality_start_activation"])
    topology = dict(cast(Mapping[str, Any], activation["quality_execution_topology"]))
    return contract.selected_worker_count(payload), {
        "quality_execution_topology": topology,
        "topology_probe": dict(cast(Mapping[str, Any], activation["topology_probe"])),
        "topology_selection": dict(cast(Mapping[str, Any], activation["topology_selection"])),
    }


def _activation_public_binding(
    path: Path,
    payload: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    envelope = cast(Mapping[str, Any], payload["attestation"])
    lock = cast(Mapping[str, Any], payload["matrix_lock_binding"])
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "experiment_id": payload["experiment_id"],
        "payload_sha256": payload["payload_sha256"],
        "attestation_mac": envelope["mac"],
        "activation_root": payload["canonical_root"],
        "matrix_lock_path": lock["path"],
        "matrix_lock_device": lock["device"],
        "matrix_lock_inode": lock["inode"],
        "base_prerequisites_sha256": payload["base_prerequisites_sha256"],
        "sealed_source_bundle_sha256": cast(
            Mapping[str, Any], payload["sealed_source_provenance"]
        ).get("bundle_sha256"),
        "sealed_launch_routing_sha256": contract.base.json_digest(payload["sealed_launch_routing"]),
        "selected_worker_count": payload["selected_worker_count"],
        "topology_probe_payload_sha256": cast(Mapping[str, Any], payload["topology_probe"])[
            "payload_sha256"
        ],
    }


def _build_activation_payload(
    *,
    prestart: PrestartQualityAuthorityV1_3_5,
    trust_root: legacy.attestation.TrustRoot,
    base_prerequisites_binding: Mapping[str, Any],
    sealed_source_provenance: Mapping[str, Any],
    sealed_launch_routing: Mapping[str, Any],
    root_identity: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    worker_count, topology = _manifest_topology(prestart.quality_context)
    base_topology = base_prerequisites_binding.get("quality_execution_topology")
    _require(
        isinstance(base_topology, Mapping)
        and cast(Mapping[str, Any], base_topology).get("worker_count") == worker_count,
        "Base prerequisites do not bind the selected v1.3.5 worker count.",
    )
    _require(
        sealed_launch_routing.get("launcher") == contract.V1_3_5_CANONICAL_GIT_OBJECT_LAUNCHER_ID
        and sealed_launch_routing.get("entrypoint_selector") == "matrix",
        "V1.3.5 sealed launch routing is invalid.",
    )
    genesis = prestart.preheldout_genesis.payload
    return legacy._attested_payload(
        {
            "schema_version": V1_3_5_ACTIVATION_SCHEMA_VERSION,
            "artifact_type": "direct-controller-v1-3-5-quality-start-activation",
            "experiment_id": contract.V1_3_5_EXPERIMENT_ID,
            "status": "activated",
            "canonical_root": root_identity["path"],
            "canonical_path": str(
                Path(cast(str, root_identity["path"]))
                / contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name
            ),
            "activation_nonce": secrets.token_hex(32),
            "quality_source": prestart.quality_context.source,
            "quality_manifest": prestart.quality_context.manifest_binding,
            "static_v1_3_4_reuse_admission": prestart.reuse_admission.public_binding,
            "static_v1_3_4_preheldout_genesis": (prestart.preheldout_genesis.public_binding),
            "prestart_absence_witness": prestart.absence_witness,
            "base_prerequisites_binding": dict(base_prerequisites_binding),
            "base_prerequisites_sha256": contract.base.json_digest(base_prerequisites_binding),
            "sealed_source_provenance": dict(sealed_source_provenance),
            "sealed_launch_routing": dict(sealed_launch_routing),
            "activation_root_identity": dict(root_identity),
            "matrix_lock_binding": dict(matrix_lock_binding),
            "selected_worker_count": worker_count,
            **topology,
            "expected_shards": genesis["expected_shards"],
            "coordinate_digest": genesis["coordinate_digest"],
            "exact_fill_arm_names": list(genesis["exact_fill_arm_names"]),
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
            "scientific_subprocesses_started_during_activation": 0,
        },
        trust_root=trust_root,
        purpose=contract.V1_3_5_QUALITY_START_ACTIVATION_ATTESTATION_PURPOSE,
    )


def _validate_activation(
    *,
    quality_context: QualityContext,
    trust_root: legacy.attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    storage_root: Path,
    canonical_root: Path,
    sealed_source_provenance: Mapping[str, Any] | None,
    sealed_launch_routing: Mapping[str, Any] | None,
    expected_base_prerequisites_binding: Mapping[str, Any] | None,
    expected_public_binding: Mapping[str, Any] | None,
    static_bundle: tuple[ValidatedReuseAdmission, ValidatedPreheldoutGenesis] | None = None,
) -> ValidatedQualityStartActivationV1_3_5:
    assert_quality_context_unchanged(quality_context)
    root_identity, lock_binding, path = _activation_root_members(
        storage_root,
        canonical_root=canonical_root,
    )
    payload, raw = _canonical_payload(path, label="V1.3.5 quality-start activation")
    legacy._verify_attested_payload(
        payload,
        trust_root=trust_root,
        purpose=contract.V1_3_5_QUALITY_START_ACTIVATION_ATTESTATION_PURPOSE,
        label="V1.3.5 quality-start activation",
    )
    reuse, genesis = (
        _static_v1_3_4_bundle(
            quality_context=quality_context,
            trust_root=trust_root,
            expected_shards=expected_shards,
            coordinate_digest=coordinate_digest,
            exact_fill_arm_names=exact_fill_arm_names,
        )
        if static_bundle is None
        else static_bundle
    )
    worker_count, topology = _manifest_topology(quality_context)
    zero_fields = (
        "completed_shards",
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "active_claim_count",
        "worker_ledger_count",
        "top_p_quality_input_count",
        "scientific_subprocesses_started_during_activation",
    )
    _require(
        payload.get("schema_version") == V1_3_5_ACTIVATION_SCHEMA_VERSION
        and payload.get("artifact_type") == "direct-controller-v1-3-5-quality-start-activation"
        and payload.get("experiment_id") == contract.V1_3_5_EXPERIMENT_ID
        and payload.get("status") == "activated"
        and payload.get("canonical_root") == str(canonical_root)
        and payload.get("canonical_path")
        == str(canonical_root / contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name)
        and payload.get("quality_source") == quality_context.source
        and payload.get("quality_manifest") == quality_context.manifest_binding
        and payload.get("static_v1_3_4_reuse_admission") == reuse.public_binding
        and payload.get("static_v1_3_4_preheldout_genesis") == genesis.public_binding
        and payload.get("activation_root_identity") == root_identity
        and payload.get("matrix_lock_binding") == lock_binding
        and payload.get("selected_worker_count") == worker_count
        and payload.get("quality_execution_topology") == topology["quality_execution_topology"]
        and payload.get("topology_probe") == topology["topology_probe"]
        and payload.get("topology_selection") == topology["topology_selection"]
        and payload.get("expected_shards") == expected_shards
        and payload.get("coordinate_digest") == coordinate_digest
        and tuple(payload.get("exact_fill_arm_names", ())) == tuple(exact_fill_arm_names)
        and payload.get("records") == []
        and payload.get("quality_evaluation_started") is False
        and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and payload.get("quality_rng_initialized") is False
        and all(payload.get(field) == 0 for field in zero_fields),
        "V1.3.5 activation identity, topology, grid, or zero-state drifted.",
    )
    raw_base = payload.get("base_prerequisites_binding")
    raw_source = payload.get("sealed_source_provenance")
    raw_route = payload.get("sealed_launch_routing")
    _require(
        isinstance(raw_base, Mapping)
        and isinstance(raw_source, Mapping)
        and isinstance(raw_route, Mapping)
        and payload.get("base_prerequisites_sha256") == contract.base.json_digest(raw_base)
        and (
            expected_base_prerequisites_binding is None
            or dict(raw_base) == dict(expected_base_prerequisites_binding)
        )
        and (sealed_source_provenance is None or dict(raw_source) == dict(sealed_source_provenance))
        and (sealed_launch_routing is None or dict(raw_route) == dict(sealed_launch_routing)),
        "V1.3.5 activation prerequisite or sealed launch binding drifted.",
    )
    public = _activation_public_binding(
        canonical_root / contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name,
        payload,
        raw,
    )
    _require(
        expected_public_binding is None or dict(expected_public_binding) == public,
        "V1.3.5 activation public binding changed.",
    )
    return ValidatedQualityStartActivationV1_3_5(
        _seal=_ACTIVATION_SEAL,
        payload=payload,
        public_binding=public,
        quality_context=quality_context,
        reuse_admission=reuse,
        preheldout_genesis=genesis,
        consumer_coordinate=None,
        root_identity=root_identity,
        matrix_lock_binding=lock_binding,
    )


_ACTIVE_LEASE_IDENTITIES: set[tuple[int, int]] = set()
_PENDING_LEASE_IDENTITIES: set[tuple[int, int]] = set()
_LEASE_GUARD = threading.Lock()


class QualityStartActivationLeaseV1_3_5:
    __slots__ = ("_seal", "activation", "_descriptor", "_closed")

    def __init__(
        self,
        seal: object,
        activation: ValidatedQualityStartActivationV1_3_5,
        descriptor: int,
    ) -> None:
        _require(
            seal is _LEASE_SEAL
            and type(activation) is ValidatedQualityStartActivationV1_3_5
            and activation._seal is _ACTIVATION_SEAL
            and type(descriptor) is int
            and descriptor >= 0,
            "V1.3.5 activation lease construction is private.",
        )
        self._seal = seal
        self.activation = activation
        self._descriptor = descriptor
        self._closed = False
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        with _LEASE_GUARD:
            _require(
                identity not in _ACTIVE_LEASE_IDENTITIES,
                "V1.3.5 activation lease is already active.",
            )
            _ACTIVE_LEASE_IDENTITIES.add(identity)

    def fileno(self) -> int:
        self.assert_held()
        return self._descriptor

    def assert_held(self) -> None:
        _require(
            self._seal is _LEASE_SEAL and not self._closed,
            "V1.3.5 activation lease is closed or invalid.",
        )
        metadata = os.fstat(self._descriptor)
        binding = self.activation.matrix_lock_binding
        current = os.stat(cast(str, binding["path"]), follow_symlinks=False)
        identity = (metadata.st_dev, metadata.st_ino)
        with _LEASE_GUARD:
            active = identity in _ACTIVE_LEASE_IDENTITIES
        _require(
            active
            and identity == (current.st_dev, current.st_ino)
            and identity == (binding["device"], binding["inode"])
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE,
            "V1.3.5 activation lock identity changed while held.",
        )

    def close(self) -> None:
        if self._closed:
            return
        metadata = os.fstat(self._descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        try:
            self.assert_held()
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            with _LEASE_GUARD:
                _ACTIVE_LEASE_IDENTITIES.discard(identity)
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> QualityStartActivationLeaseV1_3_5:
        self.assert_held()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _remove_staging(path: Path) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) == SAFE_DIRECTORY_MODE,
        "Unsafe v1.3.5 activation staging root.",
    )
    allowed = {
        contract.V1_3_5_ACTIVATION_MATRIX_LOCK_PATH.name,
        contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name,
    }
    entries = list(path.iterdir())
    for entry in entries:
        child = os.stat(entry, follow_symlinks=False)
        _require(
            entry.name in allowed
            and stat.S_ISREG(child.st_mode)
            and child.st_uid == os.getuid()
            and child.st_nlink == 1
            and stat.S_IMODE(child.st_mode) == SAFE_FILE_MODE,
            "Unsafe v1.3.5 activation staging member.",
        )
    for entry in entries:
        entry.unlink()
    path.rmdir()


def _recover_staging(parent: Path) -> None:
    for path in sorted(parent.glob(f"{V1_3_5_ACTIVATION_STAGING_PREFIX}*")):
        _remove_staging(path)
    legacy._fsync_directory(parent)


def publish_quality_start_activation(
    *,
    prestart: PrestartQualityAuthorityV1_3_5,
    trust_root: legacy.attestation.TrustRoot,
    base_prerequisites_binding: Mapping[str, Any],
    sealed_source_provenance: Mapping[str, Any],
    sealed_launch_routing: Mapping[str, Any],
) -> QualityStartActivationLeaseV1_3_5:
    authority = _require_prestart(prestart)
    root = _absolute(
        contract.V1_3_5_ACTIVATION_ROOT,
        repository_root=authority.quality_context.repository_root,
    )
    parent = legacy._exact_path(
        root.parent,
        label="V1.3.5 activation parent",
        must_exist=True,
    )
    bootstrap = acquire_gpu_lock(
        "p2-direct-controller-v1.3.5-activation-bootstrap",
        path=V1_3_5_ACTIVATION_BOOTSTRAP_LOCK_PATH,
    )
    staging: Path | None = None
    descriptor: int | None = None
    published = False
    returned = False
    lease: QualityStartActivationLeaseV1_3_5 | None = None
    try:
        bootstrap.assert_held()
        _require(not os.path.lexists(root), "Final v1.3.5 activation root is immutable.")
        _check_absence_witness(authority.quality_context, authority.absence_witness)
        _recover_staging(parent)
        staging = parent / f"{V1_3_5_ACTIVATION_STAGING_PREFIX}{secrets.token_hex(16)}"
        os.mkdir(staging, SAFE_DIRECTORY_MODE)
        os.chmod(staging, SAFE_DIRECTORY_MODE)
        lock_path = staging / contract.V1_3_5_ACTIVATION_MATRIX_LOCK_PATH.name
        descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            SAFE_FILE_MODE,
        )
        os.fchmod(descriptor, SAFE_FILE_MODE)
        os.fsync(descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        root_stat = os.stat(staging, follow_symlinks=False)
        lock_stat = os.fstat(descriptor)
        root_binding = {
            "path": str(root),
            "device": root_stat.st_dev,
            "inode": root_stat.st_ino,
            "uid": root_stat.st_uid,
            "gid": root_stat.st_gid,
            "mode": SAFE_DIRECTORY_MODE,
            "nlink": root_stat.st_nlink,
            "persistent_inode": True,
        }
        lock_binding = {
            "path": str(root / lock_path.name),
            "semantics": V1_3_5_ACTIVATION_MATRIX_LOCK_SEMANTICS,
            "persistent_inode": True,
            "device": lock_stat.st_dev,
            "inode": lock_stat.st_ino,
            "uid": lock_stat.st_uid,
            "mode": SAFE_FILE_MODE,
            "nlink": lock_stat.st_nlink,
            "unlink_on_release": False,
        }
        payload = _build_activation_payload(
            prestart=authority,
            trust_root=trust_root,
            base_prerequisites_binding=base_prerequisites_binding,
            sealed_source_provenance=sealed_source_provenance,
            sealed_launch_routing=sealed_launch_routing,
            root_identity=root_binding,
            matrix_lock_binding=lock_binding,
        )
        legacy._write_exclusive_durable(
            staging / contract.V1_3_5_QUALITY_START_ACTIVATION_PATH.name,
            legacy.canonical_pretty_json(payload),
        )
        legacy._fsync_directory(
            staging,
            exact_mode=SAFE_DIRECTORY_MODE,
        )
        static = (authority.reuse_admission, authority.preheldout_genesis)
        staged = _validate_activation(
            quality_context=authority.quality_context,
            trust_root=trust_root,
            expected_shards=cast(int, authority.preheldout_genesis.payload["expected_shards"]),
            coordinate_digest=cast(
                str,
                authority.preheldout_genesis.payload["coordinate_digest"],
            ),
            exact_fill_arm_names=cast(
                Sequence[str],
                authority.preheldout_genesis.payload["exact_fill_arm_names"],
            ),
            storage_root=staging,
            canonical_root=root,
            sealed_source_provenance=sealed_source_provenance,
            sealed_launch_routing=sealed_launch_routing,
            expected_base_prerequisites_binding=base_prerequisites_binding,
            expected_public_binding=None,
            static_bundle=static,
        )
        _require(not os.path.lexists(root), "V1.3.5 activation appeared before rename.")
        legacy._rename_directory_noreplace(staging, root)
        published = True
        legacy._fsync_directory(parent)
        final = _validate_activation(
            quality_context=authority.quality_context,
            trust_root=trust_root,
            expected_shards=cast(int, authority.preheldout_genesis.payload["expected_shards"]),
            coordinate_digest=cast(
                str,
                authority.preheldout_genesis.payload["coordinate_digest"],
            ),
            exact_fill_arm_names=cast(
                Sequence[str],
                authority.preheldout_genesis.payload["exact_fill_arm_names"],
            ),
            storage_root=root,
            canonical_root=root,
            sealed_source_provenance=sealed_source_provenance,
            sealed_launch_routing=sealed_launch_routing,
            expected_base_prerequisites_binding=base_prerequisites_binding,
            expected_public_binding=staged.public_binding,
            static_bundle=static,
        )
        current = os.stat(final.matrix_lock_binding["path"], follow_symlinks=False)
        held = os.fstat(descriptor)
        _require(
            (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino),
            "V1.3.5 activation lock changed across atomic publication.",
        )
        lease = QualityStartActivationLeaseV1_3_5(_LEASE_SEAL, final, descriptor)
        descriptor = None
        lease.assert_held()
        returned = True
        return lease
    finally:
        try:
            if descriptor is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            if staging is not None and not published and os.path.lexists(staging):
                _remove_staging(staging)
                legacy._fsync_directory(parent)
            if not returned and lease is not None:
                lease.close()
        finally:
            bootstrap.close()
        _require(returned or not published, "Published v1.3.5 activation failed validation.")


def load_activated_quality_authority(
    *,
    quality_context: QualityContext,
    trust_root: legacy.attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    sealed_source_provenance: Mapping[str, Any] | None = None,
    sealed_launch_routing: Mapping[str, Any] | None = None,
    expected_base_prerequisites_binding: Mapping[str, Any] | None = None,
    expected_public_binding: Mapping[str, Any] | None = None,
) -> ValidatedQualityStartActivationV1_3_5:
    root = _absolute(
        contract.V1_3_5_ACTIVATION_ROOT,
        repository_root=quality_context.repository_root,
    )
    return _validate_activation(
        quality_context=quality_context,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
        storage_root=root,
        canonical_root=root,
        sealed_source_provenance=sealed_source_provenance,
        sealed_launch_routing=sealed_launch_routing,
        expected_base_prerequisites_binding=expected_base_prerequisites_binding,
        expected_public_binding=expected_public_binding,
    )


def load_activated_consumer_authority(
    *,
    quality_context: QualityContext,
    trust_root: legacy.attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    scale: str,
    training_seed: int,
    calibration_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    expected_public_binding: Mapping[str, Any] | None = None,
    sealed_source_provenance: Mapping[str, Any] | None = None,
    sealed_launch_routing: Mapping[str, Any] | None = None,
    expected_base_prerequisites_binding: Mapping[str, Any] | None = None,
) -> ActivatedConsumerAuthorityV1_3_5:
    coordinate = (scale, training_seed)
    _require(
        coordinate
        in {
            (registered_scale, registered_seed)
            for registered_scale in contract.SCALES
            for registered_seed in contract.TRAINING_SEEDS
        },
        "Activated v1.3.5 consumer coordinate is outside the admitted grid.",
    )
    full = load_activated_quality_authority(
        quality_context=quality_context,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
        sealed_source_provenance=sealed_source_provenance,
        sealed_launch_routing=sealed_launch_routing,
        expected_base_prerequisites_binding=expected_base_prerequisites_binding,
        expected_public_binding=expected_public_binding,
    )
    admitted_calibration = full.reuse_admission.calibrations[coordinate]
    admitted_checkpoint = full.reuse_admission.checkpoints[coordinate]
    _require(
        {field: admitted_calibration.public_binding[field] for field in calibration_binding}
        == dict(calibration_binding)
        and admitted_checkpoint.public_binding == dict(checkpoint_binding),
        "Activated v1.3.5 consumer artifact binding drifted.",
    )
    scoped_reuse = ValidatedReuseAdmission(
        payload=full.reuse_admission.payload,
        public_binding=full.reuse_admission.public_binding,
        calibrations={coordinate: admitted_calibration},
        checkpoints={coordinate: admitted_checkpoint},
        quality_context=full.reuse_admission.quality_context,
        execution_environment_projection=(full.reuse_admission.execution_environment_projection),
    )
    scoped_activation = ValidatedQualityStartActivationV1_3_5(
        _seal=_ACTIVATION_SEAL,
        payload=full.payload,
        public_binding=full.public_binding,
        quality_context=full.quality_context,
        reuse_admission=scoped_reuse,
        preheldout_genesis=full.preheldout_genesis,
        consumer_coordinate=coordinate,
        root_identity=full.root_identity,
        matrix_lock_binding=full.matrix_lock_binding,
    )
    return require_activated_consumer_authority(
        ActivatedConsumerAuthorityV1_3_5(
            _seal=_CONSUMER_SEAL,
            activation=scoped_activation,
            reuse_admission=scoped_reuse,
            preheldout_genesis=full.preheldout_genesis,
            coordinate=coordinate,
        )
    )


def require_activated_consumer_authority(
    consumer: ActivatedConsumerAuthorityV1_3_5,
) -> ActivatedConsumerAuthorityV1_3_5:
    _require(
        type(consumer) is ActivatedConsumerAuthorityV1_3_5
        and consumer._seal is _CONSUMER_SEAL
        and type(consumer.activation) is ValidatedQualityStartActivationV1_3_5
        and consumer.activation._seal is _ACTIVATION_SEAL
        and consumer.activation.consumer_coordinate == consumer.coordinate
        and consumer.activation.reuse_admission is consumer.reuse_admission
        and consumer.activation.preheldout_genesis is consumer.preheldout_genesis
        and set(consumer.reuse_admission.calibrations) == {consumer.coordinate}
        and set(consumer.reuse_admission.checkpoints) == {consumer.coordinate},
        "Activated v1.3.5 consumer authority is raw or escaped its coordinate.",
    )
    assert_quality_context_unchanged(consumer.activation.quality_context)
    return consumer


def revalidate_activated_quality_authority(
    activation: ValidatedQualityStartActivationV1_3_5,
    *,
    trust_root: legacy.attestation.TrustRoot,
) -> ValidatedQualityStartActivationV1_3_5:
    _require(
        type(activation) is ValidatedQualityStartActivationV1_3_5
        and activation._seal is _ACTIVATION_SEAL,
        "V1.3.5 activation authority is raw or invalid.",
    )
    payload = activation.payload
    root = Path(cast(str, payload["canonical_root"]))
    return _validate_activation(
        quality_context=activation.quality_context,
        trust_root=trust_root,
        expected_shards=cast(int, payload["expected_shards"]),
        coordinate_digest=cast(str, payload["coordinate_digest"]),
        exact_fill_arm_names=cast(Sequence[str], payload["exact_fill_arm_names"]),
        storage_root=root,
        canonical_root=root,
        sealed_source_provenance=cast(
            Mapping[str, Any],
            payload["sealed_source_provenance"],
        ),
        sealed_launch_routing=cast(
            Mapping[str, Any],
            payload["sealed_launch_routing"],
        ),
        expected_base_prerequisites_binding=cast(
            Mapping[str, Any],
            payload["base_prerequisites_binding"],
        ),
        expected_public_binding=activation.public_binding,
        static_bundle=(activation.reuse_admission, activation.preheldout_genesis),
    )


def acquire_quality_start_activation_lease(
    activation: ValidatedQualityStartActivationV1_3_5,
    *,
    trust_root: legacy.attestation.TrustRoot,
) -> QualityStartActivationLeaseV1_3_5:
    authority = revalidate_activated_quality_authority(
        activation,
        trust_root=trust_root,
    )
    path = Path(cast(str, authority.matrix_lock_binding["path"]))
    descriptor = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    acquired = False
    identity: tuple[int, int] | None = None
    try:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        _require(
            identity
            == (
                authority.matrix_lock_binding["device"],
                authority.matrix_lock_binding["inode"],
            ),
            "V1.3.5 activation lock differs from its receipt.",
        )
        with _LEASE_GUARD:
            _require(
                identity not in _ACTIVE_LEASE_IDENTITIES
                and identity not in _PENDING_LEASE_IDENTITIES,
                "V1.3.5 activation lock reentry is forbidden.",
            )
            _PENDING_LEASE_IDENTITIES.add(identity)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = True
        revalidated = revalidate_activated_quality_authority(
            authority,
            trust_root=trust_root,
        )
        lease = QualityStartActivationLeaseV1_3_5(
            _LEASE_SEAL,
            revalidated,
            descriptor,
        )
        descriptor = -1
        return lease
    finally:
        if identity is not None:
            with _LEASE_GUARD:
                _PENDING_LEASE_IDENTITIES.discard(identity)
        if descriptor >= 0:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def load_activated_reuse_admission(
    activation: ValidatedQualityStartActivationV1_3_5,
    *,
    trust_root: legacy.attestation.TrustRoot,
) -> ValidatedReuseAdmission:
    return revalidate_activated_quality_authority(
        activation,
        trust_root=trust_root,
    ).reuse_admission


def load_activated_preheldout_genesis(
    activation: ValidatedQualityStartActivationV1_3_5,
    *,
    trust_root: legacy.attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> ValidatedPreheldoutGenesis:
    authority = revalidate_activated_quality_authority(
        activation,
        trust_root=trust_root,
    )
    _require(
        authority.payload["expected_shards"] == expected_shards
        and authority.payload["coordinate_digest"] == coordinate_digest
        and tuple(authority.payload["exact_fill_arm_names"]) == tuple(exact_fill_arm_names),
        "V1.3.5 activated genesis registration drifted.",
    )
    return authority.preheldout_genesis
