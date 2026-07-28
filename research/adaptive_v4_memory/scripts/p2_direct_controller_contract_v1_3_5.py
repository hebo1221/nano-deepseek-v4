from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import p2_direct_controller_contract_v1_3 as base
from p2_direct_controller_contract_v1_3 import *  # noqa: F403

V1_3_5_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-v1.3.5"
V1_3_5_MANIFEST_STATUS = (
    "frozen_v1_3_5_same_gpu_topology_storage_amendment_after_v1_3_4_zero_quality_lineage"
)
V1_3_5_MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-exact-fill-v1-3-5.json"
)
V1_3_5_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-exact-fill-v1-3-5"
)
V1_3_5_MATRIX_SUMMARY_PATH = V1_3_5_OUTPUT_ROOT / base.MATRIX_SUMMARY_NAME
V1_3_5_INTEGRITY_OUTPUT_PATH = (
    V1_3_5_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-5.integrity.json"
)
V1_3_5_SUMMARY_OUTPUT_PATH = V1_3_5_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-5.summary.json"
# The authenticated calibration/checkpoint inventory remains the exact,
# read-only v1.3.4 predecessor pair.  No v1.3.5 copy or re-attestation exists.
V1_3_5_ADMISSION_ROOT = base.V1_3_4_ADMISSION_ROOT
V1_3_5_REUSE_ADMISSION_PATH = base.V1_3_4_REUSE_ADMISSION_PATH
V1_3_5_PREHELDOUT_GENESIS_PATH = base.V1_3_4_PREHELDOUT_GENESIS_PATH
V1_3_5_ACTIVATION_ROOT = V1_3_5_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-5-activation"
V1_3_5_ACTIVATION_MATRIX_LOCK_PATH = V1_3_5_ACTIVATION_ROOT / "matrix.lock"
V1_3_5_QUALITY_START_ACTIVATION_PATH = V1_3_5_ACTIVATION_ROOT / "quality-start-activation.json"
V1_3_5_WORKER_LEDGER_ROOT = V1_3_5_OUTPUT_ROOT.parent / (
    f".{V1_3_5_OUTPUT_ROOT.name}.p2-direct-controller-workers-v1-3-5"
)
V1_3_5_PERSISTENT_SESSION_LEDGER_ROOT = V1_3_5_OUTPUT_ROOT.parent / (
    f".{V1_3_5_OUTPUT_ROOT.name}.p2-direct-controller-persistent-sessions-v1-3-5"
)
V1_3_5_PERSISTENT_SESSION_LEDGER_LOCK_PATH = (
    V1_3_5_PERSISTENT_SESSION_LEDGER_ROOT.parent
    / f"{V1_3_5_PERSISTENT_SESSION_LEDGER_ROOT.name}.lock"
)

V1_3_5_SHARD_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-shard-v1.3.5"
V1_3_5_MATRIX_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-matrix-v1.3.5"
V1_3_5_WORKER_LEDGER_EXPERIMENT_ID = (
    "p2-post-rank-direct-controller-exact-fill-worker-ledger-v1.3.5"
)
V1_3_5_INTEGRITY_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-integrity-v1.3.5"
V1_3_5_SUMMARY_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-summary-v1.3.5"
V1_3_5_CANONICAL_GIT_OBJECT_LAUNCHER_ID = "p2-direct-controller-git-object-launcher-v1-3-5"
V1_3_5_QUALITY_START_ACTIVATION_ATTESTATION_PURPOSE = "p2-direct-v1.3.5-quality-start-activation-v1"
V1_3_5_SHARD_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-shard-v1-3-5"
V1_3_5_MATRIX_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-matrix-v1-3-5"
V1_3_5_WORKER_LEDGER_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-worker-ledger-v1-3-5"
V1_3_5_INTEGRITY_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-integrity-v1-3-5"
V1_3_5_SUMMARY_ATTESTATION_PURPOSE = "p2-direct-controller-exact-fill-summary-v1-3-5"
V1_3_5_PERSISTENT_SESSION_PLAN_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-5-persistent-session-plan-v1"
)
V1_3_5_PERSISTENT_SESSION_WORK_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-5-persistent-session-work-v1"
)
V1_3_5_PERSISTENT_SESSION_RESULT_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-5-persistent-session-result-v1"
)
V1_3_5_PERSISTENT_SESSION_RECEIPT_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-5-persistent-session-receipt-v1"
)
V1_3_5_PERSISTENT_SESSION_LAUNCH_LEDGER_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-5-persistent-launch-ledger-v1"
)
V1_3_5_PERSISTENT_SESSION_TERMINAL_LEDGER_ATTESTATION_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-5-persistent-terminal-ledger-v1"
)
V1_3_5_READY_ONLY_PREFLIGHT_SESSION_ROLE = base.V1_3_4_READY_ONLY_PREFLIGHT_SESSION_ROLE
V1_3_5_QUALITY_SESSION_ROLE = base.V1_3_4_QUALITY_SESSION_ROLE
# These remain per persistent evaluator.  The manifest separately binds the
# aggregate normal-path bound as 1 + (10 * selected_worker_count).
V1_3_5_QUALITY_SESSION_NORMAL_PATH_MODEL_LOADS = base.V1_3_4_QUALITY_SESSION_NORMAL_PATH_MODEL_LOADS
V1_3_5_READY_ONLY_PREFLIGHT_MODEL_LOADS = base.V1_3_4_READY_ONLY_PREFLIGHT_MODEL_LOADS
V1_3_5_TOTAL_NORMAL_PATH_CHECKPOINT_MODEL_LOADS = (
    base.V1_3_4_TOTAL_NORMAL_PATH_CHECKPOINT_MODEL_LOADS
)

V1_3_4_STATIC_MANIFEST_SHA256 = "571a3eb28d442fda3bdc33d2129ba578dd7bcc54c7e870b6372d22757ebc7566"
V1_3_4_STATIC_MANIFEST_BYTES = 103_485
V1_3_4_STATIC_ADMISSION_SHA256 = "80671b6b7344bbc12afbb1029f75f6ab9b3c8c45f9d8b0f7c26251daee70a8b6"
V1_3_4_STATIC_ADMISSION_BYTES = 88_361
V1_3_4_STATIC_GENESIS_SHA256 = "e3b1947b2e333a794aa41c7c695491a392f18d8bb8db4cd8ca785166e90acc47"
V1_3_4_STATIC_GENESIS_BYTES = 87_925

V1_3_5_IMPLEMENTATION_PATHS = (
    *base.V1_3_4_IMPLEMENTATION_PATHS,
    "research/adaptive_v4_memory/scripts/p2_direct_controller_contract_v1_3_5.py",
    "research/adaptive_v4_memory/scripts/p2_direct_controller_topology_v1_3_5.py",
    "research/adaptive_v4_memory/scripts/probe_p2_direct_controller_topology_v1_3_5.py",
)

TOPOLOGY_PROBE_BINDING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "bytes",
        "payload_sha256",
        "attestation_mac",
        "candidate_worker_counts",
        "selected_worker_count",
        "semantic_equivalence_passed",
        "quality_values_accessed",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parent_manifest_payload() -> dict[str, Any]:
    path = base.V1_3_4_MANIFEST_PATH
    raw = path.read_bytes()
    _require(
        len(raw) == V1_3_4_STATIC_MANIFEST_BYTES
        and hashlib.sha256(raw).hexdigest() == V1_3_4_STATIC_MANIFEST_SHA256,
        "Static v1.3.4 predecessor manifest bytes drifted.",
    )
    payload = json.loads(raw.decode("utf-8"))
    _require(isinstance(payload, dict), "Static v1.3.4 predecessor manifest is invalid.")
    return cast(dict[str, Any], payload)


def _validate_probe_binding(
    binding: Mapping[str, Any], *, selected_worker_count: int
) -> dict[str, Any]:
    checked = dict(binding)
    _require(
        set(checked) == TOPOLOGY_PROBE_BINDING_FIELDS
        and isinstance(checked.get("path"), str)
        and base.is_sha256(checked.get("sha256"))
        and type(checked.get("bytes")) is int
        and cast(int, checked["bytes"]) > 0
        and base.is_sha256(checked.get("payload_sha256"))
        and base.is_sha256(checked.get("attestation_mac"))
        and checked.get("candidate_worker_counts") == [1, 2, 3, 4]
        and checked.get("selected_worker_count") == selected_worker_count
        and checked.get("semantic_equivalence_passed") is True
        and checked.get("quality_values_accessed") is False,
        "Topology probe binding is invalid.",
    )
    return checked


def build_v1_3_5_manifest_payload(
    *,
    attestation_key_id: str,
    implementation_tree_digest: str,
    implementation_source_commit: str,
    selected_worker_count: int,
    topology_probe_binding: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        type(selected_worker_count) is int and 1 <= selected_worker_count <= 4,
        "Selected same-GPU worker count is invalid.",
    )
    _require(base.is_sha256(attestation_key_id), "Attestation key ID is invalid.")
    _require(base.is_sha256(implementation_tree_digest), "Implementation digest is invalid.")
    _require(
        base.is_git_oid(implementation_source_commit),
        "Implementation source commit is invalid.",
    )
    probe = _validate_probe_binding(
        topology_probe_binding,
        selected_worker_count=selected_worker_count,
    )
    payload = copy.deepcopy(_parent_manifest_payload())
    payload["experiment_id"] = V1_3_5_EXPERIMENT_ID
    payload["status"] = V1_3_5_MANIFEST_STATUS
    payload["attestation"] = base.attestation.public_manifest_contract(attestation_key_id)
    payload["implementation"] = {
        "paths": list(V1_3_5_IMPLEMENTATION_PATHS),
        "source_commit": implementation_source_commit,
        "tree_digest": implementation_tree_digest,
    }
    namespaces = cast(dict[str, Any], payload["artifact_namespaces"])
    namespaces.update(
        {
            "activation_matrix_lock_path": str(V1_3_5_ACTIVATION_MATRIX_LOCK_PATH),
            "activation_root": str(V1_3_5_ACTIVATION_ROOT),
            "admission_root": str(V1_3_5_ADMISSION_ROOT),
            "integrity_output_path": str(V1_3_5_INTEGRITY_OUTPUT_PATH),
            "matrix_summary_path": str(V1_3_5_MATRIX_SUMMARY_PATH),
            "output_root": str(V1_3_5_OUTPUT_ROOT),
            "persistent_session_ledger_lock_path": str(V1_3_5_PERSISTENT_SESSION_LEDGER_LOCK_PATH),
            "persistent_session_ledger_root": str(V1_3_5_PERSISTENT_SESSION_LEDGER_ROOT),
            "preheldout_genesis_path": str(V1_3_5_PREHELDOUT_GENESIS_PATH),
            "quality_start_activation_path": str(V1_3_5_QUALITY_START_ACTIVATION_PATH),
            "reuse_admission_path": str(V1_3_5_REUSE_ADMISSION_PATH),
            "summary_output_path": str(V1_3_5_SUMMARY_OUTPUT_PATH),
            "worker_ledger_root": str(V1_3_5_WORKER_LEDGER_ROOT),
            "v1_3_4_quality_output_namespace_reused": False,
            "v1_3_4_static_admission_namespace_reused_read_only": True,
        }
    )
    namespaces["attestation_purposes"] = {
        "integrity": V1_3_5_INTEGRITY_ATTESTATION_PURPOSE,
        "matrix": V1_3_5_MATRIX_ATTESTATION_PURPOSE,
        "persistent_session_launch_ledger": (
            V1_3_5_PERSISTENT_SESSION_LAUNCH_LEDGER_ATTESTATION_PURPOSE
        ),
        "persistent_session_plan": V1_3_5_PERSISTENT_SESSION_PLAN_ATTESTATION_PURPOSE,
        "persistent_session_receipt": (V1_3_5_PERSISTENT_SESSION_RECEIPT_ATTESTATION_PURPOSE),
        "persistent_session_result": V1_3_5_PERSISTENT_SESSION_RESULT_ATTESTATION_PURPOSE,
        "persistent_session_terminal_ledger": (
            V1_3_5_PERSISTENT_SESSION_TERMINAL_LEDGER_ATTESTATION_PURPOSE
        ),
        "persistent_session_work": V1_3_5_PERSISTENT_SESSION_WORK_ATTESTATION_PURPOSE,
        "preheldout_genesis": base.V1_3_4_PREHELDOUT_GENESIS_ATTESTATION_PURPOSE,
        "quality_start_activation": V1_3_5_QUALITY_START_ACTIVATION_ATTESTATION_PURPOSE,
        "reuse_admission": base.V1_3_4_REUSE_ADMISSION_ATTESTATION_PURPOSE,
        "shard": V1_3_5_SHARD_ATTESTATION_PURPOSE,
        "summary": V1_3_5_SUMMARY_ATTESTATION_PURPOSE,
        "worker_ledger": V1_3_5_WORKER_LEDGER_ATTESTATION_PURPOSE,
    }
    namespaces["experiment_ids"] = {
        "integrity": V1_3_5_INTEGRITY_EXPERIMENT_ID,
        "matrix": V1_3_5_MATRIX_EXPERIMENT_ID,
        "shard": V1_3_5_SHARD_EXPERIMENT_ID,
        "summary": V1_3_5_SUMMARY_EXPERIMENT_ID,
        "worker_ledger": V1_3_5_WORKER_LEDGER_EXPERIMENT_ID,
    }
    disclosure = cast(dict[str, Any], payload["lineage_and_adaptation_disclosure"])
    disclosure["protocol_relation_to_v1_3"] = (
        "topology-and-storage-only-same-gpu-parallel-amendment-after-byte-exact-v1.3.4-"
        "zero-quality-lineage;scientific-grid-arms-estimands-and-success-gates-unchanged"
    )
    disclosure["amendment_trigger"] = (
        "single-worker-gb10-underutilization-with-preregistered-quality-blind-topology-probe"
    )
    disclosure["scientific-grid-arm-estimand-or-success-gate_changed"] = False
    disclosure["quality_outcome_used_to_create_fork"] = False
    disclosure["v1_3_4_static_predecessor"] = {
        "manifest": {
            "path": str(base.V1_3_4_MANIFEST_PATH),
            "sha256": V1_3_4_STATIC_MANIFEST_SHA256,
            "bytes": V1_3_4_STATIC_MANIFEST_BYTES,
        },
        "reuse_admission": {
            "path": str(base.V1_3_4_REUSE_ADMISSION_PATH),
            "sha256": V1_3_4_STATIC_ADMISSION_SHA256,
            "bytes": V1_3_4_STATIC_ADMISSION_BYTES,
        },
        "preheldout_genesis": {
            "path": str(base.V1_3_4_PREHELDOUT_GENESIS_PATH),
            "sha256": V1_3_4_STATIC_GENESIS_SHA256,
            "bytes": V1_3_4_STATIC_GENESIS_BYTES,
        },
        "reuse_mode": "read-only-static-predecessor-no-v1.3.4-activation-reuse",
    }
    execution = cast(dict[str, Any], payload["execution_contract"])
    execution["v1_3_5_quality_manifest_context_required"] = True
    execution["v1_3_4_quality_manifest_context_required"] = False
    execution["v1_3_4_static_predecessor_context_required"] = True
    execution["v1_3_5_ready_only_preflight_required_before_first_claim"] = True
    sealed = cast(dict[str, Any], execution["sealed_launch_and_persistent_session"])
    sealed["canonical_git_object_launcher_id"] = V1_3_5_CANONICAL_GIT_OBJECT_LAUNCHER_ID
    sealed["quality_session_normal_path_model_loads"] = 10 * selected_worker_count
    sealed["total_normal_path_checkpoint_model_loads"] = 1 + 10 * selected_worker_count
    activation = cast(dict[str, Any], sealed["quality_start_activation"])
    activation["canonical_v1_3_4_entrypoint_status"] = "retired-static-predecessor-only"
    activation["quality_execution_topology"] = {
        "distributed_execution_supported": selected_worker_count > 1,
        "device_topology": "one-physical-gpu-one-supervisor-owned-scheduler-and-device-guard",
        "worker_count": selected_worker_count,
        "worker_indices": list(range(selected_worker_count)),
        "assignment_rule": "canonical-coordinate-index-modulo-worker-count-v1",
        "same_gpu_borrowed_lease_views": True,
        "coordinator_only_publication": True,
    }
    activation["topology_probe"] = probe
    activation["matrix_lock_semantics"] = (
        "activation-root-precreated-inode-flock-plus-process-thread-mutex-v2"
    )
    activation["fresh_execution_sequence"] = [
        "validate-byte-exact-v1.3.4-static-predecessor",
        "validate-preregistered-quality-blind-topology-probe-and-selected-worker-count",
        "publish-fresh-v1.3.5-quality-start-activation",
        "run-zero-work-persistent-ready-only-preflight-on-worker-zero",
        "publish-initial-zero-record-distributed-matrix",
        "run-canonical-worker-zero-cell-only",
        "verify-only-integrity-schema-device-closed-world-and-topology-properties",
        "resume-all-selected-workers-regardless-of-first-cell-outcome-direction",
    ]
    activation["full_resume_gate"] = (
        "authenticated-exact-one-canonical-prefix-from-worker-zero-with-integrity-schema-"
        "device-closed-world-and-topology-validation"
    )
    return payload


def _index_entries(paths: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    output = subprocess.run(
        ["git", "ls-files", "-s", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parsed: list[tuple[str, str]] = []
    for entry in (line for line in output.splitlines() if line):
        metadata, separator, path = entry.partition("\t")
        fields = metadata.split()
        _require(
            separator == "\t"
            and len(fields) == 3
            and fields[2] == "0"
            and fields[0] in {"100644", "100755"}
            and base.is_git_oid(fields[1]),
            "V1.3.5 implementation index entry is invalid.",
        )
        parsed.append((path, entry))
    _validate_implementation_entries(parsed)
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(
        not tuple(path for path in untracked.splitlines() if path),
        "Untracked files exist inside the v1.3.5 implementation inventory.",
    )
    return tuple(parsed)


def _validate_implementation_entries(
    parsed: list[tuple[str, str]],
) -> tuple[str, ...]:
    _require(
        len(V1_3_5_IMPLEMENTATION_PATHS) == len(set(V1_3_5_IMPLEMENTATION_PATHS)),
        "V1.3.5 implementation roots contain duplicates.",
    )
    canonical = base._v1_3_4_validate_implementation_inventory(parsed)
    tracked = {path for path, _entry in parsed}
    missing_roots = [
        root
        for root in V1_3_5_IMPLEMENTATION_PATHS
        if root not in tracked
        and not any(path.startswith(root.rstrip("/") + "/") for path in tracked)
    ]
    _require(
        not missing_roots,
        f"V1.3.5 implementation roots are missing: {missing_roots}",
    )
    return canonical


def v1_3_5_implementation_tree_digest(
    paths: tuple[str, ...] = V1_3_5_IMPLEMENTATION_PATHS,
) -> str:
    _require(
        paths == V1_3_5_IMPLEMENTATION_PATHS,
        "Implementation path inventory drifted from the v1.3.5 contract.",
    )
    parsed = _index_entries(paths)
    canonical = _validate_implementation_entries(list(parsed))
    return base._implementation_index_digest(
        paths,
        canonical,
    )


def v1_3_5_implementation_file_paths(
    paths: tuple[str, ...] = V1_3_5_IMPLEMENTATION_PATHS,
) -> tuple[str, ...]:
    _require(
        paths == V1_3_5_IMPLEMENTATION_PATHS,
        "Implementation path inventory drifted from the v1.3.5 contract.",
    )
    return tuple(path for path, _entry in _index_entries(paths))


def v1_3_5_implementation_tree_digest_at_commit(source_commit: str) -> str:
    _require(base.is_git_oid(source_commit), "V1.3.5 source commit is invalid.")
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    _require(commit_check.returncode == 0, "V1.3.5 source commit is not a commit object.")
    output = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--full-tree",
            source_commit,
            "--",
            *V1_3_5_IMPLEMENTATION_PATHS,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parsed: list[tuple[str, str]] = []
    for entry in (line for line in output.splitlines() if line):
        metadata, separator, path = entry.partition("\t")
        fields = metadata.split()
        _require(
            separator == "\t"
            and len(fields) == 3
            and fields[1] == "blob"
            and fields[0] in {"100644", "100755"}
            and base.is_git_oid(fields[2]),
            "V1.3.5 implementation commit-tree entry is invalid.",
        )
        parsed.append((path, f"{fields[0]} {fields[2]} 0\t{path}"))
    canonical = _validate_implementation_entries(parsed)
    return base._implementation_index_digest(
        V1_3_5_IMPLEMENTATION_PATHS,
        canonical,
    )


def validate_v1_3_5_manifest_payload(
    payload: dict[str, Any], *, verify_implementation: bool = True
) -> dict[str, Any]:
    implementation = payload.get("implementation")
    execution = payload.get("execution_contract")
    _require(
        isinstance(implementation, Mapping) and isinstance(execution, Mapping),
        "V1.3.5 manifest implementation or execution contract is missing.",
    )
    checked_implementation = cast(Mapping[str, Any], implementation)
    checked_execution = cast(Mapping[str, Any], execution)
    sealed = checked_execution.get("sealed_launch_and_persistent_session")
    _require(isinstance(sealed, Mapping), "V1.3.5 sealed execution contract is missing.")
    checked_sealed = cast(Mapping[str, Any], sealed)
    activation = checked_sealed.get("quality_start_activation")
    _require(isinstance(activation, Mapping), "V1.3.5 activation contract is missing.")
    checked_activation = cast(Mapping[str, Any], activation)
    topology = checked_activation.get("quality_execution_topology")
    probe = checked_activation.get("topology_probe")
    attestation_contract = payload.get("attestation")
    _require(
        isinstance(topology, Mapping)
        and isinstance(probe, Mapping)
        and isinstance(attestation_contract, Mapping),
        "V1.3.5 topology, probe, or attestation binding is missing.",
    )
    checked_topology = cast(Mapping[str, Any], topology)
    checked_probe = cast(Mapping[str, Any], probe)
    checked_attestation = cast(Mapping[str, Any], attestation_contract)
    worker_count = checked_topology.get("worker_count")
    _require(type(worker_count) is int, "V1.3.5 selected worker count is invalid.")
    expected = build_v1_3_5_manifest_payload(
        attestation_key_id=cast(str, checked_attestation.get("key_id")),
        implementation_tree_digest=cast(str, checked_implementation.get("tree_digest")),
        implementation_source_commit=cast(str, checked_implementation.get("source_commit")),
        selected_worker_count=cast(int, worker_count),
        topology_probe_binding=checked_probe,
    )
    _require(payload == expected, "V1.3.5 manifest differs from its canonical builder.")
    if verify_implementation:
        source_commit = cast(str, checked_implementation["source_commit"])
        tree_digest = cast(str, checked_implementation["tree_digest"])
        _require(
            v1_3_5_implementation_tree_digest() == tree_digest
            and v1_3_5_implementation_tree_digest_at_commit(source_commit) == tree_digest,
            "V1.3.5 live or committed implementation differs from the manifest.",
        )
    return payload


def load_v1_3_5_manifest(
    path: Path = V1_3_5_MANIFEST_PATH, *, verify_implementation: bool = True
) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), "V1.3.5 manifest is absent or unsafe.")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    _require(
        isinstance(payload, dict)
        and raw == base.canonical_pretty_manifest_bytes(cast(Mapping[str, Any], payload)),
        "V1.3.5 manifest bytes are not canonical pretty JSON.",
    )
    return validate_v1_3_5_manifest_payload(
        cast(dict[str, Any], payload),
        verify_implementation=verify_implementation,
    )


def selected_worker_count(manifest: Mapping[str, Any]) -> int:
    execution = cast(Mapping[str, Any], manifest["execution_contract"])
    sealed = cast(Mapping[str, Any], execution["sealed_launch_and_persistent_session"])
    activation = cast(Mapping[str, Any], sealed["quality_start_activation"])
    topology = cast(Mapping[str, Any], activation["quality_execution_topology"])
    value = topology["worker_count"]
    _require(type(value) is int and 1 <= value <= 4, "Manifest worker count is invalid.")
    return cast(int, value)
