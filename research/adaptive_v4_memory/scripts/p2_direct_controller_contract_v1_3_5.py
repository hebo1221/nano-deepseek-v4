from __future__ import annotations

import copy
import functools
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import p2_direct_controller_contract_v1_3 as base
from p2_direct_controller_contract_v1_3 import *  # noqa: F403

V1_3_5_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-v1.3.5"
V1_3_5_MANIFEST_STATUS = "frozen_v1_3_5_mixed_device_block_site"
V1_3_5_MIXED_DEVICE_SITE = "rtx4090"
V1_3_5_MIXED_DEVICE_SITES = ("gb10", "rtx4090")
V1_3_5_MIXED_ASSIGNMENT_RULE = (
    "latin-rotated-four-gb10-six-rtx4090-per-full-stratum-then-site-index-modulo-worker-count-v1"
)
V1_3_5_MIXED_SITE_COORDINATE_COUNTS = {"gb10": 3_600, "rtx4090": 5_400}
# This branch freezes the exact quality-blind RTX 4090 execution projection.
V1_3_5_MIXED_EXECUTION_ENVIRONMENT_PROJECTION: Mapping[str, Any] | None = {
    "schema_version": 1,
    "python_implementation": "CPython",
    "python_version": "3.13.9",
    "python_executable": "/home/dilab/miniconda3/bin/python3.13",
    "torch_version": "2.9.1+cu128",
    "cuda_runtime_version": "12.8",
    "cuda_driver_version": "555.42.06",
    "platform_system": "Linux",
    "platform_release": "5.15.0-139-generic",
    "platform_machine": "x86_64",
    "platform_string": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31",
    "selected_device_class": {
        "name": "NVIDIA GeForce RTX 4090",
        "compute_capability": [8, 9],
        "total_memory_bytes": 25_280_184_320,
    },
}
V1_3_5_MANIFEST_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-exact-fill-v1-3-5.json"
)
V1_3_5_SUPERSEDED_ZERO_QUALITY_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-exact-fill-v1-3-5"
)
V1_3_5_SUPERSEDED_ZERO_QUALITY_ACTIVATION_ROOT = (
    V1_3_5_SUPERSEDED_ZERO_QUALITY_OUTPUT_ROOT.parent
    / "controller-exact-fill-v1-3-5-activation"
)
V1_3_5_SUPERSEDED_ZERO_QUALITY_PERSISTENT_SESSION_ROOT = (
    V1_3_5_SUPERSEDED_ZERO_QUALITY_OUTPUT_ROOT.parent
    / ".controller-exact-fill-v1-3-5.p2-direct-controller-persistent-sessions-v1-3-5"
)
V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    "controller-exact-fill-v1-3-5-parallel-retry-1"
)
V1_3_5_SUPERSEDED_BINDING_SCHEMA_ACTIVATION_ROOT = (
    V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT.parent
    / "controller-exact-fill-v1-3-5-parallel-retry-1-activation"
)
V1_3_5_SUPERSEDED_BINDING_SCHEMA_WORKER_ROOT = (
    V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT.parent
    / ".controller-exact-fill-v1-3-5-parallel-retry-1."
    "p2-direct-controller-workers-v1-3-5"
)
V1_3_5_SUPERSEDED_BINDING_SCHEMA_PERSISTENT_SESSION_ROOT = (
    V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT.parent
    / ".controller-exact-fill-v1-3-5-parallel-retry-1."
    "p2-direct-controller-persistent-sessions-v1-3-5"
)
V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    "controller-exact-fill-v1-3-5-parallel-retry-2"
)
V1_3_5_SUPERSEDED_ARM_SEMANTICS_ACTIVATION_ROOT = (
    V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT.parent
    / "controller-exact-fill-v1-3-5-parallel-retry-2-activation"
)
V1_3_5_SUPERSEDED_ARM_SEMANTICS_WORKER_ROOT = (
    V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT.parent
    / ".controller-exact-fill-v1-3-5-parallel-retry-2."
    "p2-direct-controller-workers-v1-3-5"
)
V1_3_5_SUPERSEDED_ARM_SEMANTICS_PERSISTENT_SESSION_ROOT = (
    V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT.parent
    / ".controller-exact-fill-v1-3-5-parallel-retry-2."
    "p2-direct-controller-persistent-sessions-v1-3-5"
)
V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    "controller-exact-fill-v1-3-5-parallel-final"
)
V1_3_5_SUPERSEDED_PERSISTENT_RESET_ACTIVATION_ROOT = (
    V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT.parent
    / "controller-exact-fill-v1-3-5-parallel-final-activation"
)
V1_3_5_SUPERSEDED_PERSISTENT_RESET_WORKER_ROOT = (
    V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT.parent
    / ".controller-exact-fill-v1-3-5-parallel-final."
    "p2-direct-controller-workers-v1-3-5"
)
V1_3_5_SUPERSEDED_PERSISTENT_RESET_SESSION_ROOT = (
    V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT.parent
    / ".controller-exact-fill-v1-3-5-parallel-final."
    "p2-direct-controller-persistent-sessions-v1-3-5"
)
V1_3_5_SUPERSEDED_PERSISTENT_RESET_SESSION_LOCK = (
    V1_3_5_SUPERSEDED_PERSISTENT_RESET_SESSION_ROOT.parent
    / f"{V1_3_5_SUPERSEDED_PERSISTENT_RESET_SESSION_ROOT.name}.lock"
)
V1_3_5_SUPERSEDED_TOCTOU_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    "controller-exact-fill-v1-3-5-parallel-final-2"
)
V1_3_5_FINAL_3_BRIDGE_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    "controller-exact-fill-v1-3-5-parallel-final-3"
)
V1_3_5_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    f"controller-exact-fill-v1-3-5-mixed-{V1_3_5_MIXED_DEVICE_SITE}-retry-2"
)
V1_3_5_MATRIX_SUMMARY_PATH = V1_3_5_OUTPUT_ROOT / base.MATRIX_SUMMARY_NAME
V1_3_5_INTEGRITY_OUTPUT_PATH = (
    V1_3_5_OUTPUT_ROOT.parent / f"{V1_3_5_OUTPUT_ROOT.name}.integrity.json"
)
V1_3_5_SUMMARY_OUTPUT_PATH = (
    V1_3_5_OUTPUT_ROOT.parent / f"{V1_3_5_OUTPUT_ROOT.name}.summary.json"
)
# The authenticated calibration/checkpoint inventory remains the exact,
# read-only v1.3.4 predecessor pair.  No v1.3.5 copy or re-attestation exists.
V1_3_5_ADMISSION_ROOT = base.V1_3_4_ADMISSION_ROOT
V1_3_5_REUSE_ADMISSION_PATH = base.V1_3_4_REUSE_ADMISSION_PATH
V1_3_5_PREHELDOUT_GENESIS_PATH = base.V1_3_4_PREHELDOUT_GENESIS_PATH
V1_3_5_ACTIVATION_ROOT = (
    V1_3_5_OUTPUT_ROOT.parent / f"{V1_3_5_OUTPUT_ROOT.name}-activation"
)
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
V1_3_5_SUPERSEDED_PERSISTENT_RESET_INVENTORY_SHA256 = (
    "08b41cec4569aafa2f7d99c71e9b891756ce6abc637254e2ff4805c40b949032"
)
V1_3_5_SUPERSEDED_PERSISTENT_RESET_DIRECTORY_COUNT = 14
V1_3_5_SUPERSEDED_PERSISTENT_RESET_FILE_COUNT = 45
V1_3_5_SUPERSEDED_PERSISTENT_RESET_TOTAL_BYTES = 4_500_389

V1_3_5_IMPLEMENTATION_PATHS = (
    *base.V1_3_4_IMPLEMENTATION_PATHS,
    "research/adaptive_v4_memory/reports/2026-07-28-p2-mixed-device-block-amendment.md",
    "research/adaptive_v4_memory/scripts/freeze_p2_mixed_device_manifest_v1_3_5.py",
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

V1_3_5_USER_DIRECTED_PARALLEL_OVERRIDE = {
    "probe_selected_worker_count": 1,
    "execution_worker_count": 3,
    "selection_authority": "explicit-user-directive-for-mixed-device-block-execution",
    "probe_recommendation_overridden": True,
    "measured_speedup_over_single_worker_claimed": False,
    "directive": "run-full17-as-balanced-gb10-and-rtx4090-device-blocks-with-three-workers-per-site",
}


def v1_3_5_execution_environment_projection(
    static_projection: Mapping[str, Any],
) -> dict[str, Any]:
    _require(
        V1_3_5_MIXED_DEVICE_SITE in V1_3_5_MIXED_DEVICE_SITES,
        "Mixed-device site is not registered.",
    )
    selected = (
        static_projection
        if V1_3_5_MIXED_EXECUTION_ENVIRONMENT_PROJECTION is None
        else V1_3_5_MIXED_EXECUTION_ENVIRONMENT_PROJECTION
    )
    checked = copy.deepcopy(dict(selected))
    device = checked.get("selected_device_class")
    _require(
        checked.get("schema_version") == 1
        and isinstance(device, Mapping)
        and isinstance(cast(Mapping[str, Any], device).get("name"), str)
        and isinstance(checked.get("python_version"), str)
        and isinstance(checked.get("torch_version"), str)
        and isinstance(checked.get("cuda_runtime_version"), str),
        "Mixed-device execution-environment projection is invalid.",
    )
    return checked


@functools.cache
def v1_3_5_mixed_site_coordinates(site: str) -> tuple[dict[str, int | str], ...]:
    """Return the quality-blind device block in canonical site-local order."""

    _require(site in V1_3_5_MIXED_DEVICE_SITES, "Mixed-device site is not registered.")
    scale_index = {value: index for index, value in enumerate(base.SCALES)}
    seed_index = {value: index for index, value in enumerate(base.TRAINING_SEEDS)}
    budget_index = {value: index for index, value in enumerate(base.BUDGETS)}
    family_index = {value: index for index, value in enumerate(base.FAMILIES)}
    context_index = {value: index for index, value in enumerate(base.CONTEXTS)}
    replicate_index = {value: index for index, value in enumerate(base.REPLICATES)}
    selected: list[dict[str, int | str]] = []
    for raw in base.quality_coordinates():
        offset = (
            scale_index[cast(str, raw["scale"])]
            + seed_index[cast(int, raw["training_seed"])]
            + 5 * budget_index[cast(str, raw["budget"])]
            + family_index[cast(str, raw["family"])]
            + 2 * context_index[cast(int, raw["context"])]
        ) % len(base.REPLICATES)
        is_gb10 = (
            replicate_index[cast(int, raw["replicate"])] - offset
        ) % len(base.REPLICATES) < 4
        if (site == "gb10") == is_gb10:
            selected.append(dict(raw))
    _require(
        len(selected) == V1_3_5_MIXED_SITE_COORDINATE_COUNTS[site],
        "Mixed-device site cardinality drifted.",
    )
    return tuple(selected)


def superseded_zero_quality_parallel_attempt() -> dict[str, Any]:
    session_root = V1_3_5_SUPERSEDED_ZERO_QUALITY_PERSISTENT_SESSION_ROOT
    session_nonce = "08aaf2ba81f195194b1f5bbb0ced3b0f9254f4e67508de1dc5297b7ceda07f01"
    return {
        "lineage_type": "signed-superseded-zero-quality-parallel-launch-failure",
        "manifest_commit": "69bd328d08a1df2bcdb0a01f6db93923a503e551",
        "implementation_source_commit": "01b82111a0c6e3d6fc70062f3815c841ffd7e1fb",
        "manifest": {
            "path": str(V1_3_5_MANIFEST_PATH),
            "sha256": "87e4f5ecc197cc9aff1e6575323d9128cb281dfc98efd09b842aaa63247e9a3a",
            "bytes": 106_426,
        },
        "activation": {
            "path": str(
                V1_3_5_SUPERSEDED_ZERO_QUALITY_ACTIVATION_ROOT
                / "quality-start-activation.json"
            ),
            "sha256": "ff924b355215cc433c4080b926ebf362c4c9778d53036a7ba72e7fd026c96c4f",
            "bytes": 302_045,
            "payload_sha256": (
                "d1d9a4b67f6136f1c60f7f9b06a850d2873b3acac436597e77b93b1f49b48325"
            ),
            "attestation_mac": (
                "6b96286126129f390c78ebdd135245dc978ed35659837d5b7bdddab6b88e1e75"
            ),
        },
        "matrix": {
            "path": str(
                V1_3_5_SUPERSEDED_ZERO_QUALITY_OUTPUT_ROOT / base.MATRIX_SUMMARY_NAME
            ),
            "sha256": "b33b1b94537dc1cf1a997ecc32609fc5c38047fb2df52bdd67b1824ec0932b4c",
            "bytes": 332_444,
            "payload_sha256": (
                "09e43ec7ca42e7539cfb19eec402512b5e4badf0fcebbb116d8acbabe9d622dc"
            ),
            "attestation_mac": (
                "ec9be6a03e843485938f67379efa925cc0615584a1d275f55f32a0e1925f2b15"
            ),
        },
        "persistent_ready_only_session": {
            "launch": {
                "path": str(session_root / f"{session_nonce}.launch.json"),
                "sha256": (
                    "c08eec37a689c989e7a95e6dcaaf9af73f41c3df8a8d77bd49e75a932edff50a"
                ),
                "bytes": 22_020,
                "payload_sha256": (
                    "09d127004e4ae32e6920249e0ebff9cb7626339e153967924ae4f64500d6666e"
                ),
                "attestation_mac": (
                    "4bf99ca9bbe7c6d0a72e7d115626901098b698e21910d9402b0dbf2c106d7dbf"
                ),
            },
            "terminal": {
                "path": str(session_root / f"{session_nonce}.terminal.json"),
                "sha256": (
                    "98593477741ebd05227398345ad41ecc04c92d9b2638c759fdc537dbca58572d"
                ),
                "bytes": 22_581,
                "payload_sha256": (
                    "d4ab37c892e8731f5a1428c6b7153ad676c3eb957b662384289005de35c89d59"
                ),
                "attestation_mac": (
                    "8978f15f18fd13a1b37b44bb601dfe23a32efe7c2fc597db151746367ab0fafd"
                ),
            },
        },
        "quality_state": {
            "completed_shards": 0,
            "canonical_prefix_shards": 0,
            "globally_committed_shards": 0,
            "records": [],
            "worker_ledger_count": 0,
            "gpu_worker_lease_count": 0,
            "quality_evaluation_started": False,
            "evaluation_inputs_materialized": 0,
            "quality_predictions_materialized": 0,
            "quality_outcomes_materialized": 0,
            "quality_aggregates_materialized": 0,
            "outcome_selection_performed": False,
        },
        "failure": {
            "stage": "distributed-zero-ledger-validation-before-first-worker-ledger-or-cell-claim",
            "exception": "KeyError: 0",
            "cause": (
                "ready-only preflight validation indexed an intentionally empty distributed "
                "GPU worker registry before the live supervisor fallback was wired"
            ),
        },
        "retry": {
            "output_root": str(V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT),
            "activation_root": str(V1_3_5_SUPERSEDED_BINDING_SCHEMA_ACTIVATION_ROOT),
            "scientific_grid_arm_estimand_or_success_gate_changed": False,
            "quality_outcome_used_to_configure_retry": False,
        },
    }


def superseded_unpublished_binding_schema_attempt() -> dict[str, Any]:
    session_root = V1_3_5_SUPERSEDED_BINDING_SCHEMA_PERSISTENT_SESSION_ROOT
    session_nonce = "2ecd492752e5d04338ea0102dd5e8e162965e0ac8610f7f127faf36e1ba6491c"
    claim_path = (
        V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT
        / "s55/seed-6071406/2x/single-remote-retrieval/context-80/replicate-0"
        / ".p2-direct-controller-exact-fill-v1-3-5-cell.claim"
    )
    return {
        "lineage_type": "signed-superseded-unpublished-volatile-quality-binding-schema-failure",
        "manifest_commit": "73f61887d8011d485fa007f70e01de4eb0c4902f",
        "implementation_source_commit": "58c17ac404acf671b556b911fbe149c4cae97576",
        "manifest": {
            "path": str(V1_3_5_MANIFEST_PATH),
            "sha256": "2c6d51449f9f5bc63cfed959a4b9fc4cbb043b4db8c4c11f584e78cf695b7d82",
            "bytes": 111_119,
        },
        "activation": {
            "path": str(
                V1_3_5_SUPERSEDED_BINDING_SCHEMA_ACTIVATION_ROOT
                / "quality-start-activation.json"
            ),
            "sha256": "b9e2089ce6711e34fb72b6c6f712122782d35ab24bb147343c85aab9a2aff4d7",
            "bytes": 314_744,
            "payload_sha256": (
                "56bf134d0b3cec07c482116bc2e8cc7a4ea1af5dcb3568aaf3781be77e939cbf"
            ),
            "attestation_mac": (
                "58ad761f77d80c0b6b2b78f728ea8377bdf2c74410fc458140823630525efa19"
            ),
        },
        "matrix": {
            "path": str(
                V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT / base.MATRIX_SUMMARY_NAME
            ),
            "sha256": "9e08f863b846ad7a68bf55a57e67070b5e03e99e79c3b25bbfbf2661f965da51",
            "bytes": 370_726,
            "payload_sha256": (
                "afc3295d312793da74776a2c29174559981db011f6b630c7b6bb09a27d460600"
            ),
            "attestation_mac": (
                "75765f2ab290361672cbf58f32022849e8bc6393b828182a181d2079a33e8467"
            ),
        },
        "worker_zero_ledger": {
            "path": str(
                V1_3_5_SUPERSEDED_BINDING_SCHEMA_WORKER_ROOT
                / "worker-00000-of-00003.summary.json"
            ),
            "sha256": "007dd54db28255f4193e1cbae1afe909ad430ba1479e8aef4f48170790d1e615",
            "bytes": 161_066,
            "payload_sha256": (
                "29019dba1d66537b7883319f6c7befd05669d83a8d8edef367d19401549f0d03"
            ),
            "attestation_mac": (
                "a69642a381c165e84a731b1174b4ced8820235d982b616c8a70a3b7dac5ca7ba"
            ),
            "completed_shards": 0,
            "records": [],
        },
        "quality_session": {
            "launch": {
                "path": str(session_root / f"{session_nonce}.launch.json"),
                "sha256": (
                    "5b8310769bd69b7b7305b9bd276c344c2326bb6066721b3d2df5a9144c29cfb6"
                ),
                "bytes": 22_058,
                "payload_sha256": (
                    "e912e113311d4031037fbcc2a7f4f9dafca70c74acb8e97fb5f884456a2ce810"
                ),
                "attestation_mac": (
                    "eb00b1e50f91d9f6da208d4693bd5a76f4dd0e84132b329aea064d7b5c91a5a0"
                ),
            },
            "terminal": {
                "path": str(session_root / f"{session_nonce}.terminal.json"),
                "sha256": (
                    "af936d5235e091b324c7e636a86830ede925483f0bff49396748f8f770ad022a"
                ),
                "bytes": 21_195,
                "payload_sha256": (
                    "45ae69dff823e5325b1200e4d5422d86ec3a267df4eb4dc21bc619167ee225f9"
                ),
                "attestation_mac": (
                    "25af6516cebf55661a63bccc6077f60ea52d31ce78855e49df7120343da406f7"
                ),
                "status": "child_eof",
                "child_process_returncode": 1,
                "completed_work_count": 0,
                "completed_result_count": 0,
            },
        },
        "dead_claim": {
            "path": str(claim_path),
            "sha256": "36949b7f062f41bdcf588a714e2ff7f0bf7affa386d995aeac9d1fa8460379cb",
            "bytes": 526,
            "worker_index": 0,
            "worker_count": 3,
        },
        "durable_quality_state": {
            "completed_shards": 0,
            "canonical_prefix_shards": 0,
            "globally_committed_shards": 0,
            "matrix_records": [],
            "worker_records": [],
            "published_bundle_count": 0,
            "published_envelope_count": 0,
            "published_sidecar_count": 0,
            "orphan_claim_count": 1,
        },
        "volatile_execution_disclosure": {
            "quality_evaluator_started": True,
            "quality_computation_may_have_completed_in_memory": True,
            "quality_values_read_by_supervisor_or_retry_decision": False,
        },
        "failure": {
            "stage": "first-cell-bundle-validation-before-atomic-publication",
            "exception": "ValueError: Quality-start activation public binding schema drifted.",
            "cause": (
                "the v1.3.5 producer emitted selected-worker and topology-probe bindings "
                "while the generated evaluator retained two v1.3.4 lineage field names"
            ),
        },
        "retry": {
            "output_root": str(V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT),
            "activation_root": str(V1_3_5_SUPERSEDED_ARM_SEMANTICS_ACTIVATION_ROOT),
            "scientific_grid_arm_estimand_or_success_gate_changed": False,
            "quality_values_used_to_configure_retry": False,
        },
    }


def superseded_unpublished_arm_semantics_attempt() -> dict[str, Any]:
    session_root = V1_3_5_SUPERSEDED_ARM_SEMANTICS_PERSISTENT_SESSION_ROOT
    quality_session_nonce = (
        "8d249f1336f06bfea74c11c2dd355818de401a2d2b876ee561a51d744beed757"
    )
    ready_session_nonce = (
        "b715e8a7506bddeb1892da7ed377b1867ca4e328a748238f264112924b225763"
    )
    claim_path = (
        V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT
        / "s55/seed-6071406/2x/single-remote-retrieval/context-80/replicate-0"
        / ".p2-direct-controller-exact-fill-v1-3-5-cell.claim"
    )
    return {
        "lineage_type": "signed-superseded-unpublished-arm-semantics-schema-failure",
        "manifest_commit": "6bff60145b484b111a613ba086237be8ccb2ce8d",
        "implementation_source_commit": "8e057bc1fb9909a3bb1dbf536356cf3ce79f9357",
        "manifest": {
            "path": str(V1_3_5_MANIFEST_PATH),
            "sha256": "a2079b96078ba67aaef7d8d91c055508ba9a968323db420aefee248a8da7d85b",
            "bytes": 117_196,
        },
        "activation": {
            "path": str(
                V1_3_5_SUPERSEDED_ARM_SEMANTICS_ACTIVATION_ROOT
                / "quality-start-activation.json"
            ),
            "sha256": "bbbf0d34d391e0fa06407085a3eb29a1569f03c6007f237c3cf7c22a8987af46",
            "bytes": 330_952,
            "payload_sha256": (
                "eb384467a092ae083aa7769288524456a296952ff228805584f992414b0ec6d8"
            ),
            "attestation_mac": (
                "54f4c8e678951db2ebf56f6f2b63557da3bc2d515a0818885eaf36c7252dd36b"
            ),
        },
        "matrix": {
            "path": str(
                V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT / base.MATRIX_SUMMARY_NAME
            ),
            "sha256": "c5ae40e210fda6e7fc5681bee0470090abbfb3fe5928a677e1f283167ed6fc8d",
            "bytes": 386_934,
            "payload_sha256": (
                "e84f8db1da19055a198e06b8dcb41254b90149cefd12fe134aad4d4f00f7af23"
            ),
            "attestation_mac": (
                "fd918371543b0cbd8e003d582d900d5cbeb237a698dfd15631b861e640cf6099"
            ),
        },
        "worker_zero_ledger": {
            "path": str(
                V1_3_5_SUPERSEDED_ARM_SEMANTICS_WORKER_ROOT
                / "worker-00000-of-00003.summary.json"
            ),
            "sha256": "14944e87830628ec60ee2cc6f89ccfe6f09b4384cf726f98607ea15df0ef0222",
            "bytes": 169_170,
            "payload_sha256": (
                "9801d1f97a9c95ff78644af5a9f91fb6a7e12dfe2b22f61469d106f90193ec3e"
            ),
            "attestation_mac": (
                "1a5009dfa133624b4269541e67e981ff579fb3b897e18399ddac5d8e63c56753"
            ),
            "completed_shards": 0,
            "records": [],
        },
        "ready_only_preflight": {
            "launch": {
                "path": str(session_root / f"{ready_session_nonce}.launch.json"),
                "sha256": (
                    "d6cacc5ec5c2819d39bcd05b9cf88009fe68c5f8c6f6dd9b75bccd8b470052b9"
                ),
                "bytes": 22_071,
                "payload_sha256": (
                    "f8b456b2f8822e7ed521df755e029471e53854496664bc9ae409aab5f5199e22"
                ),
                "attestation_mac": (
                    "6cde8cb06b874966c03c9151d923bc040d97354417eca25684f6ea9831d7c16a"
                ),
            },
            "terminal": {
                "path": str(session_root / f"{ready_session_nonce}.terminal.json"),
                "sha256": (
                    "06d836b88ecab51ef509969f9a3613676ac5e2d461428b8b7978a665592855a2"
                ),
                "bytes": 22_615,
                "payload_sha256": (
                    "74d4e96cc7320990ae372e694f90ecb6780727c5431a20a8bc7af38edefa2689"
                ),
                "attestation_mac": (
                    "4c7cb2a91a4a3f10d5caa093c27e3bf70d0d1101ca52f9fe1343f35aab284858"
                ),
                "status": "stopped",
                "child_process_returncode": 0,
            },
        },
        "quality_session": {
            "launch": {
                "path": str(session_root / f"{quality_session_nonce}.launch.json"),
                "sha256": (
                    "2a5d5880e4d70429916c989143463530743eca5a7bce39e2749e4982cc99f07a"
                ),
                "bytes": 22_058,
                "payload_sha256": (
                    "87928d8d7c87e729833efd1b1d1c36b3703985ef6995704f76bc7789f9edbe66"
                ),
                "attestation_mac": (
                    "287259c55daee56bbf0ab722cc1468478fb24f401f3c4cb17bd9bdc1c5051f71"
                ),
            },
            "terminal": {
                "path": str(session_root / f"{quality_session_nonce}.terminal.json"),
                "sha256": (
                    "095f507866f20a9c7c6bf37a8fceae5556033fca15a8106167e0d07b66f6217b"
                ),
                "bytes": 21_195,
                "payload_sha256": (
                    "41658b36bfb98bba30a20b61a3ea073b712cfcec714973d18a293516c2d19020"
                ),
                "attestation_mac": (
                    "3622f9cf400ab176df8a429dea78be264fc8e6dec7c40a16521086e256df4ac5"
                ),
                "status": "child_eof",
                "child_process_returncode": 1,
                "completed_work_count": 0,
                "completed_result_count": 0,
            },
        },
        "dead_claim": {
            "path": str(claim_path),
            "sha256": "6c1109af7098919562de3167f8c2c9029104d47562e7e0948495aad9f0299f54",
            "bytes": 526,
            "worker_index": 0,
            "worker_count": 3,
        },
        "durable_quality_state": {
            "completed_shards": 0,
            "canonical_prefix_shards": 0,
            "globally_committed_shards": 0,
            "matrix_records": [],
            "worker_records": [],
            "published_bundle_count": 0,
            "published_envelope_count": 0,
            "published_sidecar_count": 0,
            "orphan_claim_count": 1,
        },
        "volatile_execution_disclosure": {
            "quality_evaluator_started": True,
            "quality_computation_completed_before_bundle_validation": True,
            "quality_values_read_by_supervisor_or_retry_decision": False,
        },
        "failure": {
            "stage": "first-cell-bundle-validation-before-atomic-publication",
            "exception": "ValueError: Arm semantics drifted.",
            "cause": (
                "the producer serialized tuple-valued signal_weights as a JSON list while "
                "the generated validator compared the reopened row against the raw dataclass tuple"
            ),
        },
        "retry": {
            "output_root": str(V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT),
            "activation_root": str(V1_3_5_SUPERSEDED_PERSISTENT_RESET_ACTIVATION_ROOT),
            "scientific_grid_arm_estimand_or_success_gate_changed": False,
            "quality_values_used_to_configure_retry": False,
        },
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _superseded_persistent_reset_inventory() -> dict[str, Any]:
    roots = (
        V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT,
        V1_3_5_SUPERSEDED_PERSISTENT_RESET_ACTIVATION_ROOT,
        V1_3_5_SUPERSEDED_PERSISTENT_RESET_WORKER_ROOT,
        V1_3_5_SUPERSEDED_PERSISTENT_RESET_SESSION_ROOT,
    )
    standalone_paths = (V1_3_5_SUPERSEDED_PERSISTENT_RESET_SESSION_LOCK,)
    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for root in roots:
        _require(
            root.is_dir() and not root.is_symlink(),
            f"Superseded persistent-reset root is missing or unsafe: {root}",
        )
        directories.append(str(root))
        for path in sorted(root.rglob("*")):
            _require(
                not path.is_symlink(),
                f"Superseded persistent-reset inventory contains a symlink: {path}",
            )
            if path.is_dir():
                directories.append(str(path))
                continue
            metadata = os.stat(path, follow_symlinks=False)
            _require(
                stat.S_ISREG(metadata.st_mode),
                f"Superseded persistent-reset inventory contains a non-file: {path}",
            )
            raw = path.read_bytes()
            files.append(
                {
                    "path": str(path),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "mode": oct(stat.S_IMODE(metadata.st_mode)),
                }
            )
    for path in standalone_paths:
        metadata = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
            f"Superseded persistent-reset standalone file is unsafe: {path}",
        )
        raw = path.read_bytes()
        files.append(
            {
                "path": str(path),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "mode": oct(stat.S_IMODE(metadata.st_mode)),
            }
        )
    source = {
        "roots": [str(path) for path in roots],
        "standalone_paths": [str(path) for path in standalone_paths],
        "directories": sorted(directories),
        "files": files,
    }
    digest = base.json_digest(source)
    _require(
        len(directories) == V1_3_5_SUPERSEDED_PERSISTENT_RESET_DIRECTORY_COUNT
        and len(files) == V1_3_5_SUPERSEDED_PERSISTENT_RESET_FILE_COUNT
        and sum(cast(int, row["bytes"]) for row in files)
        == V1_3_5_SUPERSEDED_PERSISTENT_RESET_TOTAL_BYTES
        and digest == V1_3_5_SUPERSEDED_PERSISTENT_RESET_INVENTORY_SHA256,
        "Superseded persistent-reset inventory drifted.",
    )
    return {
        "roots": [str(path) for path in roots],
        "standalone_paths": [str(path) for path in standalone_paths],
        "directory_count": len(directories),
        "file_count": len(files),
        "total_bytes": sum(cast(int, row["bytes"]) for row in files),
        "inventory_sha256": digest,
    }


def superseded_persistent_reset_attempt() -> dict[str, Any]:
    output_root = V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT
    claim_path = (
        output_root
        / "s55/seed-6071406/2x/single-remote-retrieval/context-80/replicate-3"
        / ".p2-direct-controller-exact-fill-v1-3-5-cell.claim"
    )
    envelope_path = claim_path.with_name(
        "direct-s55-train-6071406-2x-single-remote-retrieval-context-80-replicate-3.json"
    )
    return {
        "lineage_type": "signed-superseded-persistent-reset-and-drain-failure",
        "manifest_commit": "a11933e9cae2eed84d795631826e2eb605a8e091",
        "implementation_source_commit": "db4d425ac61e3df49b5a44a7a0f37edf3db17870",
        "manifest": {
            "path": str(V1_3_5_MANIFEST_PATH),
            "sha256": "7a257b7a75563ca323b3bd8b12613dbb85d999f7bad48a77624b939026e63692",
            "bytes": 124_514,
        },
        "activation": {
            "path": str(
                V1_3_5_SUPERSEDED_PERSISTENT_RESET_ACTIVATION_ROOT
                / "quality-start-activation.json"
            ),
            "sha256": "8e3379036eb1356d7b199081049a033a6baf4e9b9a8af3162b5a82cc39ca0911",
            "bytes": 350_442,
            "payload_sha256": "94b05e7645128a8153696665a895527d87536c61377997ddfc13536048beb08b",
            "attestation_mac": "43eb7d013f7a864dab9f52a50b711558e6311f7603479c3ac2f6770b0a93c7b3",
        },
        "matrix": {
            "path": str(output_root / base.MATRIX_SUMMARY_NAME),
            "sha256": "8e59d2b0f9d7c9897354d639f08743ebf9240fda4eb437cf122b53888f37de87",
            "bytes": 675_527,
            "payload_sha256": "8081d42821f88f0dc17e80c5f947b42c167823c004f3537bf13b6d7d78f4b5da",
            "attestation_mac": "9056e99376abdd8fd266c2c5e275b86ea49a5454ac65d0344ef75b76640a8868",
            "status": "in_progress",
        },
        "closed_world_inventory": _superseded_persistent_reset_inventory(),
        "durable_quality_state": {
            "completed_shards": 4,
            "canonical_prefix_shards": 3,
            "globally_committed_shards": 4,
            "worker_completed_shards": [1, 2, 1],
            "committed_coordinate_keys": [
                "s55/train-6071406/2x/single-remote-retrieval/context-80/replicate-0",
                "s55/train-6071406/2x/single-remote-retrieval/context-80/replicate-1",
                "s55/train-6071406/2x/single-remote-retrieval/context-80/replicate-2",
                "s55/train-6071406/2x/single-remote-retrieval/context-80/replicate-4",
            ],
            "orphan_claim_count": 1,
            "complete_uncommitted_bundle_count": 1,
        },
        "orphan_claim": {
            "path": str(claim_path),
            "sha256": "c46072cdca8c41d1dc4869482b155d3b7637d711ce9efeb5df468feca93d6a1a",
            "bytes": 526,
            "worker_index": 0,
            "worker_count": 3,
        },
        "uncommitted_envelope": {
            "path": str(envelope_path),
            "sha256": "fc9aad825cc89748edfbb801d982eff30822122569d13f6162e12ad40ad2f4f7",
            "bytes": 37_827,
            "payload_sha256": "3c49fb5812ab0eb26147173d83a4200581fbc2d97401647394c2a7e996129d78",
            "attestation_mac": "df13353b715c85db0930c893e0fe3538b1d0ac784dbe3cc292918c8fe0535352",
        },
        "persistent_sessions": {
            "launch_count": 6,
            "terminal_count": 6,
            "child_eof_count": 4,
            "parent_crash_recovered_count": 1,
            "controlled_stop_count": 1,
            "published_bundle_reingestion_count": 4,
        },
        "failure": {
            "stage": "post-publication-persistent-reset-followed-by-concurrent-drain",
            "exception": (
                "ValueError: Persistent evaluator did not return to its model-resident "
                "allocation baseline."
            ),
            "cause": (
                "the child required byte-exact equality to the pre-first-work CUDA allocation; "
                "each quality child exited after publishing its first bundle when the "
                "post-first-work allocation was higher, and concurrent drain left one complete "
                "uncommitted bundle under a dead claim"
            ),
        },
        "retry": {
            "output_root": str(V1_3_5_FINAL_3_BRIDGE_OUTPUT_ROOT),
            "activation_root": str(
                V1_3_5_FINAL_3_BRIDGE_OUTPUT_ROOT.parent
                / f"{V1_3_5_FINAL_3_BRIDGE_OUTPUT_ROOT.name}-activation"
            ),
            "scientific_grid_arm_estimand_or_success_gate_changed": False,
            "quality_values_used_to_configure_retry": False,
            "retry_trigger_used_only_integrity_schema_device_and_process_state": True,
        },
        "quality_values_read_by_supervisor_or_retry_decision": False,
    }


def superseded_live_claim_preflight_attempt() -> dict[str, Any]:
    return {
        "lineage_type": "signed-superseded-live-claim-preflight-toctou-failure",
        "manifest_commit": "8714f45cbb296563974cb93297a1444b8e962411",
        "implementation_source_commit": "0c0f96ba36eee5db75902ba4cb7805a2b98638ad",
        "output_root": str(V1_3_5_SUPERSEDED_TOCTOU_OUTPUT_ROOT),
        "durable_state_counts": "committed=7;integrity_pass=7;integrity_fail=0;uncommitted=3;stale_claims=2",
        "failure": "coordinator closed-world scan raced live-claim bundle publication",
        "retry_output_root": str(V1_3_5_FINAL_3_BRIDGE_OUTPUT_ROOT),
        "scientific_grid_arm_estimand_or_success_gate_changed": False,
        "quality_values_read_by_supervisor_or_retry_decision": False,
    }


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


def _validate_probe_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
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
        and type(checked.get("selected_worker_count")) is int
        and 1 <= cast(int, checked["selected_worker_count"]) <= 4
        and checked.get("semantic_equivalence_passed") is True
        and checked.get("quality_values_accessed") is False,
        "Topology probe binding is invalid.",
    )
    return checked


def _topology_selection(
    *, selected_worker_count: int, probe_selected_worker_count: int
) -> dict[str, Any]:
    if selected_worker_count == probe_selected_worker_count:
        return {
            "probe_selected_worker_count": probe_selected_worker_count,
            "execution_worker_count": selected_worker_count,
            "selection_authority": "quality-blind-topology-probe",
            "probe_recommendation_overridden": False,
            "measured_speedup_over_single_worker_claimed": selected_worker_count > 1,
            "directive": None,
        }
    expected = V1_3_5_USER_DIRECTED_PARALLEL_OVERRIDE
    _require(
        probe_selected_worker_count == expected["probe_selected_worker_count"]
        and selected_worker_count == expected["execution_worker_count"],
        "Only the explicit one-to-three-worker user-directed override is admitted.",
    )
    return dict(expected)


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
    probe = _validate_probe_binding(topology_probe_binding)
    topology_selection = _topology_selection(
        selected_worker_count=selected_worker_count,
        probe_selected_worker_count=cast(int, probe["selected_worker_count"]),
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
            "final_3_bridge_output_root": str(V1_3_5_FINAL_3_BRIDGE_OUTPUT_ROOT),
            "mixed_device_site": V1_3_5_MIXED_DEVICE_SITE,
            "superseded_v1_3_5_toctou_output_root": str(V1_3_5_SUPERSEDED_TOCTOU_OUTPUT_ROOT),
            "superseded_v1_3_5_zero_quality_activation_root": str(
                V1_3_5_SUPERSEDED_ZERO_QUALITY_ACTIVATION_ROOT
            ),
            "superseded_v1_3_5_zero_quality_output_root": str(
                V1_3_5_SUPERSEDED_ZERO_QUALITY_OUTPUT_ROOT
            ),
            "superseded_v1_3_5_zero_quality_persistent_session_root": str(
                V1_3_5_SUPERSEDED_ZERO_QUALITY_PERSISTENT_SESSION_ROOT
            ),
            "superseded_v1_3_5_binding_schema_activation_root": str(
                V1_3_5_SUPERSEDED_BINDING_SCHEMA_ACTIVATION_ROOT
            ),
            "superseded_v1_3_5_binding_schema_output_root": str(
                V1_3_5_SUPERSEDED_BINDING_SCHEMA_OUTPUT_ROOT
            ),
            "superseded_v1_3_5_binding_schema_persistent_session_root": str(
                V1_3_5_SUPERSEDED_BINDING_SCHEMA_PERSISTENT_SESSION_ROOT
            ),
            "superseded_v1_3_5_binding_schema_worker_root": str(
                V1_3_5_SUPERSEDED_BINDING_SCHEMA_WORKER_ROOT
            ),
            "superseded_v1_3_5_arm_semantics_activation_root": str(
                V1_3_5_SUPERSEDED_ARM_SEMANTICS_ACTIVATION_ROOT
            ),
            "superseded_v1_3_5_arm_semantics_output_root": str(
                V1_3_5_SUPERSEDED_ARM_SEMANTICS_OUTPUT_ROOT
            ),
            "superseded_v1_3_5_arm_semantics_persistent_session_root": str(
                V1_3_5_SUPERSEDED_ARM_SEMANTICS_PERSISTENT_SESSION_ROOT
            ),
            "superseded_v1_3_5_arm_semantics_worker_root": str(
                V1_3_5_SUPERSEDED_ARM_SEMANTICS_WORKER_ROOT
            ),
            "superseded_v1_3_5_persistent_reset_activation_root": str(
                V1_3_5_SUPERSEDED_PERSISTENT_RESET_ACTIVATION_ROOT
            ),
            "superseded_v1_3_5_persistent_reset_output_root": str(
                V1_3_5_SUPERSEDED_PERSISTENT_RESET_OUTPUT_ROOT
            ),
            "superseded_v1_3_5_persistent_reset_persistent_session_root": str(
                V1_3_5_SUPERSEDED_PERSISTENT_RESET_SESSION_ROOT
            ),
            "superseded_v1_3_5_persistent_reset_worker_root": str(
                V1_3_5_SUPERSEDED_PERSISTENT_RESET_WORKER_ROOT
            ),
            "v1_3_4_quality_output_namespace_reused": False,
            "v1_3_4_static_admission_namespace_reused_read_only": True,
            "v1_3_5_superseded_zero_quality_namespace_reused": False,
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
        "device-blocked-mixed-gpu-execution-amendment-after-v1.3.5-final-3-bridge;"
        "scientific-grid-arms-estimands-and-success-gates-unchanged;"
        "device-by-arm-interactions-preregistered"
    )
    disclosure["amendment_trigger"] = (
        "explicit-user-request-to-shorten-full17-with-a-gb10-plus-rtx4090-"
        "quality-blind-balanced-device-block-design"
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
    disclosure["v1_3_5_superseded_zero_quality_parallel_attempt"] = (
        superseded_zero_quality_parallel_attempt()
    )
    disclosure["v1_3_5_superseded_unpublished_binding_schema_attempt"] = (
        superseded_unpublished_binding_schema_attempt()
    )
    disclosure["v1_3_5_superseded_unpublished_arm_semantics_attempt"] = (
        superseded_unpublished_arm_semantics_attempt()
    )
    disclosure["v1_3_5_superseded_persistent_reset_attempt"] = (
        superseded_persistent_reset_attempt()
    )
    disclosure["v1_3_5_superseded_live_claim_preflight_attempt"] = superseded_live_claim_preflight_attempt()
    execution = cast(dict[str, Any], payload["execution_contract"])
    execution["v1_3_5_quality_manifest_context_required"] = True
    execution["v1_3_4_quality_manifest_context_required"] = False
    execution["v1_3_4_static_predecessor_context_required"] = True
    execution["v1_3_5_ready_only_preflight_required_before_first_claim"] = True
    sealed = cast(dict[str, Any], execution["sealed_launch_and_persistent_session"])
    sealed["canonical_git_object_launcher_id"] = V1_3_5_CANONICAL_GIT_OBJECT_LAUNCHER_ID
    sealed["quality_session_normal_path_model_loads"] = 10 * selected_worker_count
    sealed["total_normal_path_checkpoint_model_loads"] = 1 + 10 * selected_worker_count
    sealed["persistent_model_allocation_reset"] = {
        "model_parameter_and_buffer_identity_shape_dtype_device_version_exact": True,
        "pre_first_work_cuda_allocation_recorded": True,
        "first_completed_cell_may_raise_allocation_baseline_once": True,
        "first_completed_cell_baseline_stabilization_requires_unchanged_model_state": True,
        "every_later_cell_requires_exact_stabilized_allocation_equality": True,
        "stabilization_scope": "one-persistent-child-process",
        "quality_values_used_to_set_or_validate_baseline": False,
    }
    activation = cast(dict[str, Any], sealed["quality_start_activation"])
    activation["canonical_v1_3_4_entrypoint_status"] = "retired-static-predecessor-only"
    activation["quality_execution_topology"] = {
        "distributed_execution_supported": selected_worker_count > 1,
        "device_topology": "one-physical-gpu-per-device-block-site",
        "device_block_site": V1_3_5_MIXED_DEVICE_SITE,
        "device_block_sites": list(V1_3_5_MIXED_DEVICE_SITES),
        "site_coordinate_count": V1_3_5_MIXED_SITE_COORDINATE_COUNTS[
            V1_3_5_MIXED_DEVICE_SITE
        ],
        "site_coordinate_counts": dict(V1_3_5_MIXED_SITE_COORDINATE_COUNTS),
        "replicates_per_full_stratum": {"gb10": 4, "rtx4090": 6},
        "execution_environment_projection": (
            "static-v1.3.4-predecessor"
            if V1_3_5_MIXED_EXECUTION_ENVIRONMENT_PROJECTION is None
            else copy.deepcopy(V1_3_5_MIXED_EXECUTION_ENVIRONMENT_PROJECTION)
        ),
        "worker_count": selected_worker_count,
        "worker_indices": list(range(selected_worker_count)),
        "assignment_rule": V1_3_5_MIXED_ASSIGNMENT_RULE,
        "same_gpu_borrowed_lease_views": True,
        "coordinator_only_publication": True,
    }
    activation["topology_probe"] = probe
    activation["topology_selection"] = topology_selection
    activation["matrix_lock_semantics"] = (
        "activation-root-precreated-inode-flock-plus-process-thread-mutex-v2"
    )
    activation["fresh_execution_sequence"] = [
        "validate-byte-exact-v1.3.4-static-predecessor",
        "validate-preregistered-quality-blind-topology-probe-and-record-selection-authority",
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
