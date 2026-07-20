from __future__ import annotations

import argparse
import base64
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
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
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

# Prospective v1.3.2 import-boundary-amendment namespace.  The v1.3 admission
# bundle above is immutable lineage and is never selected by path-presence.
V1_3_2_QUALITY_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-v1.3.2"
V1_3_2_MANIFEST_RELATIVE_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-exact-fill-v1-3-2.json"
)
V1_3_2_QUALITY_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/controller-exact-fill-v1-3-2"
)
V1_3_2_ADMISSION_ROOT = V1_3_2_QUALITY_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-2-admission"
V1_3_2_DEFAULT_ADMISSION_PATH = V1_3_2_ADMISSION_ROOT / "historical-reuse-admission.json"
V1_3_2_DEFAULT_GENESIS_PATH = V1_3_2_ADMISSION_ROOT / "preheldout-genesis.json"
V1_3_2_ACTIVATION_ROOT = (
    V1_3_2_QUALITY_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-2-activation"
)
V1_3_2_ACTIVATION_MATRIX_LOCK_PATH = V1_3_2_ACTIVATION_ROOT / "matrix.lock"
V1_3_2_QUALITY_START_ACTIVATION_PATH = V1_3_2_ACTIVATION_ROOT / "quality-start-activation.json"
V1_3_2_MATRIX_SUMMARY_PATH = V1_3_2_QUALITY_OUTPUT_ROOT / "controller-matrix.summary.json"
V1_3_2_INTEGRITY_OUTPUT_PATH = (
    V1_3_2_QUALITY_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-2.integrity.json"
)
V1_3_2_SUMMARY_OUTPUT_PATH = (
    V1_3_2_QUALITY_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-2.summary.json"
)
V1_3_2_WORKER_LEDGER_ROOT = V1_3_2_QUALITY_OUTPUT_ROOT.parent / (
    f".{V1_3_2_QUALITY_OUTPUT_ROOT.name}.p2-direct-controller-workers-v1-3-2"
)
V1_3_2_PERSISTENT_SESSION_LEDGER_ROOT = V1_3_2_QUALITY_OUTPUT_ROOT.parent / (
    f".{V1_3_2_QUALITY_OUTPUT_ROOT.name}.p2-direct-controller-persistent-sessions-v1-3-2"
)
V1_3_2_PERSISTENT_SESSION_LEDGER_LOCK_PATH = (
    V1_3_2_PERSISTENT_SESSION_LEDGER_ROOT.parent
    / f"{V1_3_2_PERSISTENT_SESSION_LEDGER_ROOT.name}.lock"
)
V1_3_2_PROSPECTIVE_QUALITY_PATHS = (
    V1_3_2_QUALITY_OUTPUT_ROOT,
    V1_3_2_MATRIX_SUMMARY_PATH,
    V1_3_2_WORKER_LEDGER_ROOT,
    V1_3_2_PERSISTENT_SESSION_LEDGER_ROOT,
    V1_3_2_PERSISTENT_SESSION_LEDGER_LOCK_PATH,
    V1_3_2_INTEGRITY_OUTPUT_PATH,
    V1_3_2_SUMMARY_OUTPUT_PATH,
)
V1_3_2_ADMISSION_STAGING_PREFIX = ".controller-exact-fill-v1-3-2-admission.staging-"
V1_3_2_ACTIVATION_STAGING_PREFIX = ".controller-exact-fill-v1-3-2-activation.staging-"
V1_3_2_ACTIVATION_BOOTSTRAP_LOCK_PATH = Path(
    "/tmp/adaptive-v4-direct-controller-v1-3-2-activation-bootstrap.lock"
)
V1_3_2_ACTIVATION_MATRIX_LOCK_SEMANTICS = (
    "activation-root-precreated-inode-flock-exclusive-process-owner-v1"
)
V1_3_2_CANONICAL_GIT_OBJECT_LAUNCHER_ID = "p2-direct-controller-git-object-launcher-v1-3-2"
V1_3_2_REUSE_ADMISSION_PURPOSE = "p2-direct-v1.3.2-reuse-admission-v1"
V1_3_2_PREHELDOUT_GENESIS_PURPOSE = "p2-direct-v1.3.2-preheldout-genesis-v1"
V1_3_2_QUALITY_START_ACTIVATION_PURPOSE = "p2-direct-v1.3.2-quality-start-activation-v1"

# Byte-exact v1.3 signed-empty lineage.  Validation below checks every value,
# both HMAC domains, and live absence of every superseded prospective path.
V1_3_SUPERSEDED_RESULT_SOURCE_COMMIT = "62ba095614f9614cb85a06f1a043a89a21f7f1dd"
V1_3_SUPERSEDED_RESULT_SOURCE_TREE = "5779ff8261029c1c53bc834a6dc7d16c84963fcf"
V1_3_SUPERSEDED_ATTESTATION_KEY_ID = (
    "67f433c02a291f9b1c9e65218171b6da46ef019567ee24406b4738c2ddf765bf"
)
V1_3_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT = "a729723bc3e513dd10ed06ac65f7128b4b726f98"
V1_3_SUPERSEDED_IMPLEMENTATION_SOURCE_TREE = "a7b1cc62f22d3993fc180c33444c96d0ed0eea3d"
V1_3_SUPERSEDED_IMPLEMENTATION_DIGEST = (
    "5c96c102ffa7cabc974d0a86ba60f536b513829c59ded31d615111c1312a7ebb"
)
V1_3_SUPERSEDED_MANIFEST_SHA256 = "7fb1bc578b112031ce1ad914c7ea1cdc09fdf21a2c6fcde8362e93df60dd994e"
V1_3_SUPERSEDED_MANIFEST_BYTES = 51_431
V1_3_SUPERSEDED_LIVE_IMPLEMENTATION_INVENTORY_DIGEST = (
    "e9f878efbd6228d6a2a0e55ac475946b70644362030516d9c18801d675c301cb"
)
V1_3_SUPERSEDED_LIVE_IMPLEMENTATION_FILE_COUNT = 51
V1_3_SUPERSEDED_ADMISSION_SHA256 = (
    "0ec0efc962b77389c1a2db3cd1cf96a19ae989afb131c0d8d05a242f7a38a574"
)
V1_3_SUPERSEDED_ADMISSION_BYTES = 109_919
V1_3_SUPERSEDED_ADMISSION_PAYLOAD_SHA256 = (
    "954c088184df272e6509953ea0a33e226e7d075a2da48f00b0d8af15d2f1df3a"
)
V1_3_SUPERSEDED_ADMISSION_ATTESTATION_PAYLOAD_SHA256 = (
    "4312aef8a00ce564eb4e5e394ead61037fe4afba41549848e0f18edbf41d6c73"
)
V1_3_SUPERSEDED_ADMISSION_ATTESTATION_MAC = (
    "d8a1759466fe64ac192f95c02022b5bfb6ce96c0581864e5270bfe1407201d19"
)
V1_3_SUPERSEDED_GENESIS_SHA256 = "dff6d05c69c04b1b83d7edb3d5da650daef691d7e508defcc70decd9328bbdd9"
V1_3_SUPERSEDED_GENESIS_BYTES = 4_314
V1_3_SUPERSEDED_GENESIS_PAYLOAD_SHA256 = (
    "887c47fcc55697075db6293816e9c3fd14c6f382a5f93c5b34fb82b879f44b85"
)
V1_3_SUPERSEDED_GENESIS_ATTESTATION_PAYLOAD_SHA256 = (
    "5a570eef17ba9e51ca0c92b88070847748174f3df451aaf5b4ba29ffb1923d74"
)
V1_3_SUPERSEDED_GENESIS_ATTESTATION_MAC = (
    "6b47fa8667c92a961f664793547b1991a9e87cd1866847760609e34bf2a68336"
)
V1_3_SUPERSEDED_HISTORICAL_RECEIPT_PAYLOAD_SHA256 = (
    "3af058c7c0bb18b417643af0469b733bf8987fa1a0348b1b117ac0522e241131"
)
V1_3_SUPERSEDED_CANONICAL_NONOBSERVATION_PAYLOAD_SHA256 = (
    "7c4f5a7304f87f12e1a74bb0301829f4c961e3a17c9a75dfd13883b3ba2af4b1"
)
V1_3_SUPERSEDED_LINEAGE_SHA256 = (
    "9ec45d1ededa576d76f0cec3614ce0606e195aee2a1475aa5ff1f1d7ff4f39cc"
)

# Byte-exact v1.3.1 zero-quality launch-failure lineage.  Unlike the v1.3
# signed-empty prefix above, these immutable roots include an activated
# zero-record matrix, a failed persistent launch, and one dead-owner claim.
V1_3_1_SUPERSEDED_EXPERIMENT_ID = "p2-post-rank-direct-controller-exact-fill-v1.3.1"
V1_3_1_SUPERSEDED_MATRIX_EXPERIMENT_ID = (
    "p2-post-rank-direct-controller-exact-fill-matrix-v1.3.1"
)
V1_3_1_SUPERSEDED_MANIFEST_RELATIVE_PATH = Path(
    "research/adaptive_v4_memory/manifests/"
    "p2-post-rank-direct-controller-exact-fill-v1-3-1.json"
)
V1_3_1_SUPERSEDED_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/"
    "controller-exact-fill-v1-3-1"
)
V1_3_1_SUPERSEDED_ADMISSION_ROOT = (
    V1_3_1_SUPERSEDED_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-1-admission"
)
V1_3_1_SUPERSEDED_ADMISSION_PATH = (
    V1_3_1_SUPERSEDED_ADMISSION_ROOT / "historical-reuse-admission.json"
)
V1_3_1_SUPERSEDED_GENESIS_PATH = (
    V1_3_1_SUPERSEDED_ADMISSION_ROOT / "preheldout-genesis.json"
)
V1_3_1_SUPERSEDED_ACTIVATION_ROOT = (
    V1_3_1_SUPERSEDED_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-1-activation"
)
V1_3_1_SUPERSEDED_ACTIVATION_LOCK_PATH = V1_3_1_SUPERSEDED_ACTIVATION_ROOT / "matrix.lock"
V1_3_1_SUPERSEDED_ACTIVATION_PATH = (
    V1_3_1_SUPERSEDED_ACTIVATION_ROOT / "quality-start-activation.json"
)
V1_3_1_SUPERSEDED_MATRIX_PATH = (
    V1_3_1_SUPERSEDED_OUTPUT_ROOT / "controller-matrix.summary.json"
)
V1_3_1_SUPERSEDED_CLAIM_RELATIVE_PATH = Path(
    "s55/seed-6071406/2x/single-remote-retrieval/context-80/replicate-0/"
    ".p2-direct-controller-exact-fill-v1-3-1-cell.claim"
)
V1_3_1_SUPERSEDED_CLAIM_PATH = (
    V1_3_1_SUPERSEDED_OUTPUT_ROOT / V1_3_1_SUPERSEDED_CLAIM_RELATIVE_PATH
)
V1_3_1_SUPERSEDED_SESSION_NONCE = (
    "caecf57c56d3ee51263c150cdc24ebf6a1c789dca085cf0e5eaa81a970f9d222"
)
V1_3_1_SUPERSEDED_LAUNCH_AUTHORITY_NONCE = (
    "6c64fafcbea3cb212b341adc173b4197033f097d04bf61465dc54c8e47eb97a4"
)
V1_3_1_SUPERSEDED_SESSION_ROOT = V1_3_1_SUPERSEDED_OUTPUT_ROOT.parent / (
    ".controller-exact-fill-v1-3-1.p2-direct-controller-persistent-sessions-v1-3-1"
)
V1_3_1_SUPERSEDED_SESSION_LOCK_PATH = Path(f"{V1_3_1_SUPERSEDED_SESSION_ROOT}.lock")
V1_3_1_SUPERSEDED_SESSION_LAUNCH_PATH = V1_3_1_SUPERSEDED_SESSION_ROOT / (
    f"{V1_3_1_SUPERSEDED_SESSION_NONCE}.launch.json"
)
V1_3_1_SUPERSEDED_SESSION_TERMINAL_PATH = V1_3_1_SUPERSEDED_SESSION_ROOT / (
    f"{V1_3_1_SUPERSEDED_SESSION_NONCE}.terminal.json"
)
V1_3_1_SUPERSEDED_WORKER_ROOT = V1_3_1_SUPERSEDED_OUTPUT_ROOT.parent / (
    ".controller-exact-fill-v1-3-1.p2-direct-controller-workers-v1-3-1"
)
V1_3_1_SUPERSEDED_INTEGRITY_PATH = (
    V1_3_1_SUPERSEDED_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-1.integrity.json"
)
V1_3_1_SUPERSEDED_SUMMARY_PATH = (
    V1_3_1_SUPERSEDED_OUTPUT_ROOT.parent / "controller-exact-fill-v1-3-1.summary.json"
)

V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT = "b473b237689ed7466a3a27db7054d2770afe60b8"
V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_TREE = "6d35e8458495112f4179fde0bdb0fe58cb7ad055"
V1_3_1_SUPERSEDED_IMPLEMENTATION_DIGEST = (
    "6b861b5c1869c42442cc432bf940c0403d030a31c4b0438c1bd99166a30db7e0"
)
V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT = "828e8e0c57042b71a1117cd067b113f20d61c04c"
V1_3_1_SUPERSEDED_RESULT_SOURCE_TREE = "2d0b3b4be19893c3c14ca69308db0e40850eb76a"
V1_3_1_SUPERSEDED_MANIFEST_SHA256 = "980144f5ae00ae8381f4ced1a93d614a1bdd2a361060fa344bb55db8273ae50d"
V1_3_1_SUPERSEDED_MANIFEST_BYTES = 56_967
V1_3_1_SUPERSEDED_ADMISSION_SHA256 = "c2c786d437a3d42d3329947d8d7941751d7b9561317eaf00c92e10cfeb27c5ea"
V1_3_1_SUPERSEDED_ADMISSION_BYTES = 11_092
V1_3_1_SUPERSEDED_ADMISSION_PAYLOAD_SHA256 = (
    "26d110d67658ec285730d6fc09ddb3e107a2d14b5427abe6e97ff3cc50c037ec"
)
V1_3_1_SUPERSEDED_ADMISSION_ATTESTATION_PAYLOAD_SHA256 = (
    "7fec73573a4ba5def25ff52f82f53b8e391dc326c0eacbe74c46f0450bd33b9d"
)
V1_3_1_SUPERSEDED_ADMISSION_ATTESTATION_MAC = (
    "4bdfd54ae51114a8ebfe0baebed8c2bec996023c370c076931ab505526374a98"
)
V1_3_1_SUPERSEDED_GENESIS_SHA256 = "3d8a4f0f006f916586a223ab5f0cc92e72c17e66267eb85bc8ebbaa9ef27948e"
V1_3_1_SUPERSEDED_GENESIS_BYTES = 10_427
V1_3_1_SUPERSEDED_GENESIS_PAYLOAD_SHA256 = (
    "d4da758f4be9615b3dd1a9c97eeb47e1737e4668bc6a7b59c5190624f9e5971c"
)
V1_3_1_SUPERSEDED_GENESIS_ATTESTATION_PAYLOAD_SHA256 = (
    "b496882bad92c3a3020abcd5b3f8ff4a859d87afa3335dee4cdc24ffab285609"
)
V1_3_1_SUPERSEDED_GENESIS_ATTESTATION_MAC = (
    "c657bcdf9c4f7a33ff8c6b1d43c32cad45a2e3dd349eab6b12167bc0ebfa9821"
)
V1_3_1_SUPERSEDED_ACTIVATION_LOCK_SHA256 = (
    "101d46a05395e701d04b42b471ca67dc778117e01c87405196df227f4216c642"
)
V1_3_1_SUPERSEDED_ACTIVATION_LOCK_BYTES = 509
V1_3_1_SUPERSEDED_ACTIVATION_SHA256 = "e14b9e672fe0beb93203b51faa252ec17885b60ec52eac4890cdfe89e0e151d6"
V1_3_1_SUPERSEDED_ACTIVATION_BYTES = 177_754
V1_3_1_SUPERSEDED_ACTIVATION_PAYLOAD_SHA256 = (
    "a4a28c0dba896fd74253a3bb7edd95921018999f3e2d9421924896f45233cdf3"
)
V1_3_1_SUPERSEDED_ACTIVATION_ATTESTATION_PAYLOAD_SHA256 = (
    "95fc5e73a43c658a9012f3d6cf1e8c818e246de45e2d134aa45cc0114438844c"
)
V1_3_1_SUPERSEDED_ACTIVATION_ATTESTATION_MAC = (
    "ef7c07823390abf627060974c8a5f120334ca61cd5fb1fa2bf8bdc54bbde8624"
)
V1_3_1_SUPERSEDED_MATRIX_SHA256 = "79d726212ca6bf9af1f9f8c380d9dbd8b0e98fcfc36fd6cec64870bdbe17eebb"
V1_3_1_SUPERSEDED_MATRIX_BYTES = 194_432
V1_3_1_SUPERSEDED_MATRIX_PAYLOAD_SHA256 = (
    "f25756152b9103273cc4f3710cdfbccf6c09be9e7f1289b5218616aafc1f01b1"
)
V1_3_1_SUPERSEDED_MATRIX_ATTESTATION_PAYLOAD_SHA256 = (
    "dca120aa04fa4b053a31dc862df24fb2aa45e91c058c0bac5d994b95b468c110"
)
V1_3_1_SUPERSEDED_MATRIX_ATTESTATION_MAC = (
    "f78b4d0c2760e5620a323392124ea219f7f6646b6d57c0a50f74e01cb3e720fa"
)
V1_3_1_SUPERSEDED_CLAIM_SHA256 = "4deb50235d9f6999e09a2b4e3bf3ae53f689c1a75e3b6afd54a2ce53949b0c20"
V1_3_1_SUPERSEDED_CLAIM_BYTES = 532
V1_3_1_SUPERSEDED_CLAIM_PID = 151_659
V1_3_1_SUPERSEDED_CLAIM_PROCESS_START_TICKS = 328_868_946
V1_3_1_SUPERSEDED_SESSION_LAUNCH_SHA256 = (
    "db938290daa24802a8cb084525e535b1068417f33bf8608a7a87fef635a46e9a"
)
V1_3_1_SUPERSEDED_SESSION_LAUNCH_BYTES = 21_891
V1_3_1_SUPERSEDED_SESSION_LAUNCH_PAYLOAD_SHA256 = (
    "81f3c53b5c4987392b18f0b5d756b006d03c6dc90c4e870db65e949d0580490a"
)
V1_3_1_SUPERSEDED_SESSION_LAUNCH_ATTESTATION_PAYLOAD_SHA256 = (
    "42e6e951e63090142ee86ec8fcf40e6a4074befd44dc4e2552b5dce887319a9a"
)
V1_3_1_SUPERSEDED_SESSION_LAUNCH_ATTESTATION_MAC = (
    "4137a43e35744f207960428797e527ab4b2d35ceeafb5ed7dd6f9d749f56a07d"
)
V1_3_1_SUPERSEDED_SESSION_TERMINAL_SHA256 = (
    "4a638fc41ab27fafcdd04c0bd47eea31fb7122b05442f79a4a0cdb68f3a91454"
)
V1_3_1_SUPERSEDED_SESSION_TERMINAL_BYTES = 19_659
V1_3_1_SUPERSEDED_SESSION_TERMINAL_PAYLOAD_SHA256 = (
    "932b29a7f471b2bd32a6818c9fa6c3bfdfbcec20bec4daac255c1b2b7396c310"
)
V1_3_1_SUPERSEDED_SESSION_TERMINAL_ATTESTATION_PAYLOAD_SHA256 = (
    "58e125144404540d36677eb92c20404fcdb4bb0a660eec6ada6506150b02560f"
)
V1_3_1_SUPERSEDED_SESSION_TERMINAL_ATTESTATION_MAC = (
    "5e7a743cae34b3681e076ad47b06378f3ed6217cfc86804a33f33fe50a3ac060"
)
V1_3_1_SUPERSEDED_SESSION_LOCK_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
V1_3_1_SUPERSEDED_SESSION_LOCK_BYTES = 0

V1_3_1_SUPERSEDED_REUSE_ADMISSION_PURPOSE = "p2-direct-v1.3.1-reuse-admission-v1"
V1_3_1_SUPERSEDED_PREHELDOUT_GENESIS_PURPOSE = "p2-direct-v1.3.1-preheldout-genesis-v1"
V1_3_1_SUPERSEDED_ACTIVATION_PURPOSE = "p2-direct-v1.3.1-quality-start-activation-v1"
V1_3_1_SUPERSEDED_MATRIX_PURPOSE = "p2-direct-controller-exact-fill-matrix-v1-3-1"
V1_3_1_SUPERSEDED_SESSION_PLAN_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-1-persistent-session-plan-v1"
)
V1_3_1_SUPERSEDED_SESSION_LAUNCH_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-1-persistent-launch-ledger-v1"
)
V1_3_1_SUPERSEDED_SESSION_TERMINAL_PURPOSE = (
    "p2-direct-controller-exact-fill-v1-3-1-persistent-terminal-ledger-v1"
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
V1_1_MANIFEST_RELATIVE_PATH = Path(
    "research/adaptive_v4_memory/manifests/p2-post-rank-direct-controller-v1-1.json"
)
V1_1_MANIFEST_SHA256 = "1d059f83ca73945b9df5dbee20752fbf3f99c0a24794c533be9c52a4230b4c0b"

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
HISTORICAL_SUPERSEDED_TRAINING_LEDGER = HISTORICAL_TRAINING_ROOT / "training-matrix.summary.json"
HISTORICAL_TRAINING_ADMISSION = HISTORICAL_TRAINING_ROOT / "training-v1-1-preheldout-admission.json"
HISTORICAL_CALIBRATION_LEDGER = HISTORICAL_CALIBRATION_ROOT / "calibration-matrix-v1-2.summary.json"
HISTORICAL_CALIBRATION_ADMISSION = (
    HISTORICAL_CALIBRATION_ROOT / "calibration-v1-2-retry-admission.json"
)
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
HISTORICAL_CALIBRATION_ADMISSION_BYTES = 9711
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
HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_DOMAIN = (
    "adaptive-v4-memory:p2-direct-v1.3:historical-thread-fs-isolation:semantic-observation:v1"
)
NONOBSERVATION_PURPOSE = "p2-direct-v1.3-canonical-nonobservation-v1"
REUSE_ADMISSION_PURPOSE = "p2-direct-v1.3-reuse-admission-v1"
PREHELDOUT_GENESIS_PURPOSE = "p2-direct-v1.3-preheldout-genesis-v1"
LEGACY_CALIBRATION_PURPOSE = "p2-direct-soft-lag-calibration-v1"
LEGACY_RETRY_ADMISSION_PURPOSE = "p2-direct-calibration-one-shot-retry-admission-v1.2"
MODULE_ORIGIN_AUDIT_SEMANTICS = (
    "source-only-frozen-inventory-with-sealed-admission-isolated-pycache-and-trusted-venv-v2"
)
ARCHIVED_THIRD_PARTY_RUNTIME_CLAIM = (
    "exact sealed site-packages directory identity with a trusted preinstalled environment; "
    "third-party package bytes and distribution records are not integrity-bound, and same-UID "
    "dependency mutation is outside the admission threat model"
)

RECEIPT_SCHEMA_VERSION = 2
NONOBSERVATION_SCHEMA_VERSION = 1
ADMISSION_SCHEMA_VERSION = 1
GENESIS_SCHEMA_VERSION = 1
V1_3_2_ADMISSION_SCHEMA_VERSION = 1
V1_3_2_GENESIS_SCHEMA_VERSION = 1
V1_3_2_ACTIVATION_SCHEMA_VERSION = 1
SAFE_FILE_MODE = 0o600
SAFE_DIRECTORY_MODE = 0o700
HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS = 10.0

_HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_FIELDS = (
    "python_thread_count_before_probe",
    "python_thread_count_during_probe",
    "python_thread_count_after_probe",
    "native_task_count_before_probe",
    "native_task_count_during_probe",
    "native_task_count_after_probe",
    "preexisting_non_main_native_task_count",
    "probe_thread_native_task_count",
    "main_transition",
    "probe_thread_transition",
    "preexisting_non_main_transition",
    "task_set_restored_after_probe",
    "all_tasks_detached_after_probe",
)


@dataclass(frozen=True)
class _HistoricalThreadFsIsolationCapability:
    seal: object
    repository_root: Path
    detached_root: Path
    root_identity: tuple[int, int, int, int, int]
    detached_identity: tuple[int, int, int, int, int]
    main_native_id: int
    baseline_task_ids: tuple[int, ...]
    baseline_task_cwds: tuple[tuple[int, str, tuple[int, int, int, int, int]], ...]
    claim_bytes: bytes
    claim_sha256: str


@dataclass(frozen=True)
class _HistoricalSupersededPathSpellingAuthority:
    seal: object
    capability: _HistoricalThreadFsIsolationCapability
    mode: str


@dataclass(frozen=True)
class _HistoricalCalibrationProvenanceAuthority:
    seal: object
    capability: _HistoricalThreadFsIsolationCapability
    outer_profile_sha256: str
    expected_inner_profiles: tuple[str, str]
    observed_inner_attempts: list[None]
    observed_inner_profiles: list[str]


_HISTORICAL_THREAD_FS_ISOLATION_SEAL = object()
_HISTORICAL_SUPERSEDED_PATH_SPELLING_SEAL = object()
_HISTORICAL_CALIBRATION_PROVENANCE_SEAL = object()
_HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY: _HistoricalThreadFsIsolationCapability | None = None
_HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED = False


def _new_historical_thread_fs_capability_registry() -> tuple[
    Callable[[_HistoricalThreadFsIsolationCapability], None],
    Callable[[object], bool],
    Callable[[], _HistoricalThreadFsIsolationCapability | None],
]:
    installed: _HistoricalThreadFsIsolationCapability | None = None

    def install(capability: _HistoricalThreadFsIsolationCapability) -> None:
        nonlocal installed
        _require(
            installed is None and capability.seal is _HISTORICAL_THREAD_FS_ISOLATION_SEAL,
            "Historical thread fs-isolation capability registry is already installed.",
        )
        installed = capability

    def contains(candidate: object) -> bool:
        return installed is not None and candidate is installed

    def get() -> _HistoricalThreadFsIsolationCapability | None:
        return installed

    return install, contains, get


(
    _install_historical_thread_fs_capability,
    _historical_thread_fs_capability_is_installed,
    _get_installed_historical_thread_fs_capability,
) = _new_historical_thread_fs_capability_registry()
_HISTORICAL_ROOT_CWD_AUTHORITY: ContextVar[
    tuple[
        _HistoricalThreadFsIsolationCapability,
        Path,
        Path,
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int],
    ]
    | None
] = ContextVar("adaptive_v4_historical_root_cwd_authority", default=None)
_HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY: ContextVar[
    _HistoricalSupersededPathSpellingAuthority | None
] = ContextVar(
    "adaptive_v4_historical_superseded_path_spelling_authority",
    default=None,
)
_HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY: ContextVar[
    _HistoricalCalibrationProvenanceAuthority | None
] = ContextVar(
    "adaptive_v4_historical_calibration_provenance_authority",
    default=None,
)

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
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
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


@dataclass(frozen=True)
class SupersededEmptyLineageV1_3:
    _seal: object
    manifest: dict[str, Any]
    reuse_admission: dict[str, Any]
    preheldout_genesis: dict[str, Any]
    public_binding: dict[str, Any]


@dataclass(frozen=True)
class SupersededZeroQualityFailureLineageV1_3_1:
    _seal: object
    manifest: dict[str, Any]
    reuse_admission: dict[str, Any]
    preheldout_genesis: dict[str, Any]
    quality_start_activation: dict[str, Any]
    matrix: dict[str, Any]
    persistent_launch: dict[str, Any]
    persistent_terminal: dict[str, Any]
    orphan_claim: dict[str, Any]
    public_binding: dict[str, Any]


@dataclass(frozen=True)
class PrestartQualityAuthorityV1_3_2:
    _seal: object
    quality_context: QualityContext
    reuse_admission: ValidatedReuseAdmission
    preheldout_genesis: ValidatedPreheldoutGenesis
    superseded_empty_lineage: SupersededEmptyLineageV1_3
    superseded_failure_lineage: SupersededZeroQualityFailureLineageV1_3_1
    absence_witness: dict[str, Any]


@dataclass(frozen=True)
class ValidatedQualityStartActivationV1_3_2:
    _seal: object
    payload: dict[str, Any]
    public_binding: dict[str, Any]
    quality_context: QualityContext
    reuse_admission: ValidatedReuseAdmission
    preheldout_genesis: ValidatedPreheldoutGenesis
    superseded_empty_lineage: SupersededEmptyLineageV1_3
    superseded_failure_lineage: SupersededZeroQualityFailureLineageV1_3_1
    consumer_coordinate: tuple[str, int] | None
    root_identity: dict[str, Any]
    matrix_lock_binding: dict[str, Any]


@dataclass(frozen=True)
class ActivatedConsumerAuthorityV1_3_2:
    _seal: object
    activation: ValidatedQualityStartActivationV1_3_2
    reuse_admission: ValidatedReuseAdmission
    preheldout_genesis: ValidatedPreheldoutGenesis
    coordinate: tuple[str, int]


@dataclass(frozen=True)
class _ActivatedConsumerScopeV1_3_2:
    _seal: object
    coordinate: tuple[str, int]
    calibration_binding: dict[str, Any]
    checkpoint_binding: dict[str, Any]


_SUPERSEDED_EMPTY_LINEAGE_SEAL = object()
_SUPERSEDED_ZERO_QUALITY_FAILURE_LINEAGE_V1_3_1_SEAL = object()
_PRESTART_QUALITY_AUTHORITY_V1_3_2_SEAL = object()
_VALIDATED_QUALITY_START_ACTIVATION_V1_3_2_SEAL = object()
_QUALITY_START_ACTIVATION_LEASE_V1_3_2_SEAL = object()
_ACTIVATED_CONSUMER_AUTHORITY_V1_3_2_SEAL = object()
_ACTIVATED_CONSUMER_SCOPE_V1_3_2_SEAL = object()
_ACTIVE_V1_3_2_ACTIVATION_LEASE_FDS: set[int] = set()
_PENDING_V1_3_2_ACTIVATION_LEASE_IDENTITIES: set[tuple[int, int]] = set()
_ACTIVE_V1_3_2_ACTIVATION_LEASES_GUARD = threading.Lock()


class QualityStartActivationLeaseV1_3_2:
    """Own the activation-root matrix flock without exposing any key bytes."""

    __slots__ = ("_seal", "activation", "_descriptor", "_closed")

    def __init__(
        self,
        seal: object,
        activation: ValidatedQualityStartActivationV1_3_2,
        descriptor: int,
    ) -> None:
        _require(
            seal is _QUALITY_START_ACTIVATION_LEASE_V1_3_2_SEAL
            and type(activation) is ValidatedQualityStartActivationV1_3_2
            and activation._seal is _VALIDATED_QUALITY_START_ACTIVATION_V1_3_2_SEAL
            and activation.consumer_coordinate is None
            and type(descriptor) is int
            and descriptor >= 0,
            "Quality-start activation lease construction is private.",
        )
        self._seal = seal
        self.activation = activation
        self._descriptor = descriptor
        self._closed = False
        with _ACTIVE_V1_3_2_ACTIVATION_LEASES_GUARD:
            _require(
                descriptor not in _ACTIVE_V1_3_2_ACTIVATION_LEASE_FDS,
                "Quality-start activation lease descriptor is already registered.",
            )
            _ACTIVE_V1_3_2_ACTIVATION_LEASE_FDS.add(descriptor)

    def fileno(self) -> int:
        self.assert_held()
        return self._descriptor

    def assert_held(self) -> None:
        _require(
            self._seal is _QUALITY_START_ACTIVATION_LEASE_V1_3_2_SEAL and not self._closed,
            "Quality-start activation lease is closed or invalid.",
        )
        with _ACTIVE_V1_3_2_ACTIVATION_LEASES_GUARD:
            _require(
                self._descriptor in _ACTIVE_V1_3_2_ACTIVATION_LEASE_FDS,
                "Quality-start activation lease is not registered.",
            )
        metadata = os.fstat(self._descriptor)
        binding = self.activation.matrix_lock_binding
        path = Path(cast(str, binding["path"]))
        current = os.stat(path, follow_symlinks=False)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == (current.st_dev, current.st_ino)
            and binding.get("device") == metadata.st_dev
            and binding.get("inode") == metadata.st_ino
            and metadata.st_uid == current.st_uid == os.getuid()
            and metadata.st_nlink == current.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == stat.S_IMODE(current.st_mode) == SAFE_FILE_MODE,
            "Quality-start activation matrix lock identity changed while held.",
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.assert_held()
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
        finally:
            with _ACTIVE_V1_3_2_ACTIVATION_LEASES_GUARD:
                _ACTIVE_V1_3_2_ACTIVATION_LEASE_FDS.discard(self._descriptor)
            os.close(self._descriptor)
            self._closed = True

    def __enter__(self) -> QualityStartActivationLeaseV1_3_2:
        self.assert_held()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
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
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(repository_root),
            *arguments,
        ],
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


def establish_v1_3_2_quality_context(
    manifest_path: Path,
    *,
    implementation_paths: Sequence[str],
    repository_root: Path = REPOSITORY_ROOT,
) -> QualityContext:
    """Establish only the v1.3.2 context; never auto-select by file presence."""

    root = _exact_path(repository_root, label="Repository root", must_exist=True)
    _require(root.is_dir(), "Repository root is not a directory.")
    source = _source_state(root)
    _require(source["dirty"] is False, "Quality context requires a clean result-source checkout.")
    path = _exact_path(
        manifest_path,
        label="Final v1.3.2 manifest",
        repository_root=root,
        must_exist=True,
    )
    payload, opened = _load_json_nofollow(
        path,
        label="Final v1.3.2 manifest",
        require_canonical_pretty_bytes=True,
    )
    try:
        contract = importlib.import_module("p2_direct_controller_contract_v1_3")
        raw_validator = getattr(contract, "validate_v1_3_2_manifest_payload", None)
        canonical_paths = getattr(contract, "V1_3_2_IMPLEMENTATION_PATHS", None)
        canonical_experiment_id = getattr(contract, "V1_3_2_EXPERIMENT_ID", None)
        canonical_manifest_path = getattr(contract, "V1_3_2_MANIFEST_PATH", None)
        _require(callable(raw_validator), "Canonical v1.3.2 manifest validator is unavailable.")
        _require(
            canonical_experiment_id == V1_3_2_QUALITY_EXPERIMENT_ID
            and tuple(canonical_paths or ()) == tuple(implementation_paths)
            and canonical_manifest_path == V1_3_2_MANIFEST_RELATIVE_PATH,
            "Caller boundary differs from the canonical v1.3.2 contract.",
        )
        validator = cast(Callable[..., Any], raw_validator)
        validated_manifest = validator(dict(payload), verify_implementation=True)
        _require(
            validated_manifest == payload,
            "Canonical v1.3.2 validator changed or rejected the manifest.",
        )
        _require(
            payload.get("experiment_id") == V1_3_2_QUALITY_EXPERIMENT_ID,
            "Wrong v1.3.2 experiment manifest.",
        )
        implementation = payload.get("implementation")
        _require(isinstance(implementation, Mapping), "Manifest implementation is missing.")
        paths = tuple(implementation_paths)
        implementation_map = cast(Mapping[str, Any], implementation)
        _require(
            tuple(implementation_map.get("paths", ())) == paths,
            "Manifest implementation inventory differs from the caller inventory.",
        )
        source_commit = implementation_map.get("source_commit")
        tree_digest = implementation_map.get("tree_digest")
        _require(_is_git_oid(source_commit), "Manifest implementation commit is invalid.")
        _require(_is_sha256(tree_digest), "Manifest implementation digest is invalid.")
        current_entries = _index_inventory(root, paths)
        _require(
            _implementation_index_digest(paths, current_entries) == tree_digest,
            "Current v1.3.2 implementation differs from the manifest.",
        )
        frozen_inventory = _commit_inventory(root, cast(str, source_commit), paths)
        _require(
            _commit_inventory_digest(frozen_inventory, paths) == tree_digest,
            "V1.3.2 source commit does not reproduce its tree digest.",
        )
        live_inventory = _quality_live_implementation_inventory(
            root,
            paths=paths,
            source_commit=cast(str, source_commit),
            expected_digest=cast(str, tree_digest),
        )
        _assert_ancestor(root, cast(str, source_commit), cast(str, source["commit"]))
        _assert_ancestor(
            root,
            V1_3_SUPERSEDED_RESULT_SOURCE_COMMIT,
            cast(str, source["commit"]),
        )
        public_attestation = payload.get("attestation")
        _require(isinstance(public_attestation, Mapping), "Manifest attestation is missing.")
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
            "experiment_id": V1_3_2_QUALITY_EXPERIMENT_ID,
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


def _historical_command_option(command: Any, option: str, *, label: str) -> str:
    _require(
        isinstance(command, list) and all(isinstance(item, str) for item in command),
        f"{label} command is invalid.",
    )
    positions = [index for index, item in enumerate(command) if item == option]
    _require(len(positions) == 1, f"{label} command must contain exactly one {option}.")
    position = positions[0]
    _require(position + 1 < len(command), f"{label} command has no value for {option}.")
    return cast(str, command[position + 1])


def _resolve_historical_command_relative_path(
    raw: object,
    *,
    repository_root: Path,
    expected_path: Path,
    label: str,
    require_directory: bool = True,
) -> tuple[str, Path]:
    root = _exact_path(repository_root, label="Historical command repository root", must_exist=True)
    _require(root.is_dir(), "Historical command repository root must be a directory.")
    _require(isinstance(raw, str) and bool(raw), f"{label} must be a non-empty string.")
    value = cast(str, raw)
    components = value.split("/")
    _require(
        not Path(value).is_absolute()
        and all(component not in {"", ".", ".."} for component in components),
        f"{label} must be a canonical repository-relative path without parent traversal.",
    )
    candidate = root.joinpath(*components)
    expected = _exact_path(expected_path, label=f"Expected {label}", must_exist=True)
    _require(candidate == expected, f"{label} does not name its exact expected artifact path.")
    resolved = _exact_path(candidate, label=label, must_exist=True)
    _require(
        resolved.is_dir() if require_directory else resolved.is_file(),
        f"{label} must identify an existing {'directory' if require_directory else 'regular file'}.",
    )
    return value, resolved


def _historical_training_context_profiles(
    repository_root: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    root = _exact_path(
        repository_root,
        label="Historical training-context repository root",
        must_exist=True,
    )
    current_payload, current_opened = _load_json_nofollow(
        root / HISTORICAL_TRAINING_LEDGER,
        label="Historical amended training-context ledger",
    )
    superseded_payload, superseded_opened = _load_json_nofollow(
        root / HISTORICAL_SUPERSEDED_TRAINING_LEDGER,
        label="Historical superseded training-context ledger",
    )
    admission_payload, admission_opened = _load_json_nofollow(
        root / HISTORICAL_TRAINING_ADMISSION,
        label="Historical training-context admission",
    )
    try:
        current_manifest = current_payload.get("manifest")
        current_source = current_payload.get("source")
        superseded_manifest = superseded_payload.get("manifest")
        superseded_source = superseded_payload.get("source")
        _require(
            current_opened.sha256 == HISTORICAL_TRAINING_LEDGER_SHA256
            and superseded_opened.sha256 == HISTORICAL_SUPERSEDED_TRAINING_LEDGER_SHA256
            and admission_opened.sha256 == HISTORICAL_TRAINING_ADMISSION_SHA256
            and isinstance(current_manifest, Mapping)
            and isinstance(current_source, Mapping)
            and isinstance(superseded_manifest, Mapping)
            and isinstance(superseded_source, Mapping)
            and admission_payload.get("current_manifest") == current_manifest
            and admission_payload.get("current_source") == current_source
            and admission_payload.get("superseded_manifest") == superseded_manifest
            and current_manifest.get("experiment_id") == "p2-post-rank-direct-controller-v1.1"
            and current_manifest.get("sha256") == V1_1_MANIFEST_SHA256
            and current_source == {"commit": V1_1_RESULT_SOURCE_COMMIT, "dirty": False}
            and superseded_manifest.get("experiment_id") == "p2-post-rank-direct-controller-v1"
            and _is_sha256(superseded_manifest.get("sha256"))
            and _is_sha256(superseded_manifest.get("implementation_digest"))
            and _is_git_oid(superseded_manifest.get("implementation_source_commit"))
            and isinstance(superseded_source.get("commit"), str)
            and _is_git_oid(superseded_source.get("commit"))
            and superseded_source.get("dirty") is False,
            "Historical training context profiles differ from signed admission evidence.",
        )
        profiles: dict[tuple[str, int], dict[str, Any]] = {}
        first_coordinate = (SCALES[0], TRAINING_SEEDS[0])
        for scale in SCALES:
            for seed in TRAINING_SEEDS:
                coordinate = (scale, seed)
                manifest = (
                    superseded_manifest if coordinate == first_coordinate else current_manifest
                )
                source = superseded_source if coordinate == first_coordinate else current_source
                profiles[coordinate] = {
                    "manifest_binding": dict(cast(Mapping[str, Any], manifest)),
                    "source": dict(cast(Mapping[str, Any], source)),
                }
        current_opened.assert_unchanged()
        superseded_opened.assert_unchanged()
        admission_opened.assert_unchanged()
        return profiles
    finally:
        admission_opened.close()
        superseded_opened.close()
        current_opened.close()


def _historical_training_command_path_inventory(
    repository_root: Path,
) -> tuple[dict[str, Any], ...]:
    root = _exact_path(repository_root, label="Historical command repository root", must_exist=True)
    ledger_path = root / HISTORICAL_TRAINING_LEDGER
    context_profiles = _historical_training_context_profiles(root)
    ledger, ledger_opened = _load_json_nofollow(
        ledger_path,
        label="Historical training command ledger",
    )
    try:
        runs = ledger.get("runs")
        _require(
            ledger_opened.sha256 == HISTORICAL_TRAINING_LEDGER_SHA256
            and isinstance(runs, list)
            and len(runs) == len(SCALES) * len(TRAINING_SEEDS),
            "Historical training command ledger binding or run inventory drifted.",
        )
        rows: list[dict[str, Any]] = []
        coordinates: set[tuple[str, int]] = set()
        for raw_run in cast(list[Any], runs):
            _require(isinstance(raw_run, Mapping), "Historical training run is invalid.")
            run = cast(Mapping[str, Any], raw_run)
            scale = run.get("scale")
            seed = run.get("seed")
            _require(
                scale in SCALES
                and seed in TRAINING_SEEDS
                and (cast(str, scale), cast(int, seed)) not in coordinates,
                "Historical training command coordinate is invalid or duplicated.",
            )
            coordinate = (cast(str, scale), cast(int, seed))
            coordinates.add(coordinate)
            output_dir = root / HISTORICAL_TRAINING_ROOT / coordinate[0] / f"seed-{coordinate[1]}"
            summary_path = output_dir / f"{coordinate[0]}-training.summary.json"
            summary_binding = run.get("summary")
            _require(
                isinstance(summary_binding, Mapping)
                and summary_binding.get("path") == str(summary_path)
                and _is_sha256(summary_binding.get("sha256"))
                and type(summary_binding.get("bytes")) is int
                and cast(int, summary_binding.get("bytes")) > 0,
                "Historical training summary binding is invalid.",
            )
            binding = cast(Mapping[str, Any], summary_binding)
            summary, summary_opened = _load_json_nofollow(
                summary_path,
                label=f"Historical training summary {coordinate[0]}/{coordinate[1]}",
            )
            try:
                envelope = summary.get("attestation")
                command = summary.get("command")
                _require(
                    summary_opened.sha256 == binding.get("sha256")
                    and summary_opened.bytes == binding.get("bytes")
                    and summary.get("payload_sha256") == binding.get("payload_sha256")
                    and isinstance(envelope, Mapping)
                    and envelope.get("mac") == binding.get("attestation_mac")
                    and summary.get("scale") == coordinate[0]
                    and summary.get("seed") == coordinate[1],
                    "Historical training summary differs from its exact ledger binding.",
                )
                output_dir_value = _historical_command_option(
                    command,
                    "--output-dir",
                    label=f"Historical training summary {coordinate[0]}/{coordinate[1]}",
                )
                recorded_command = list(cast(list[str], command))
                expected_executable = str(root / ARCHIVED_PYTHON_RELATIVE_PATH)
                _require(
                    recorded_command[0] == expected_executable,
                    "Historical training command executable spelling drifted.",
                )
                recorded, resolved = _resolve_historical_command_relative_path(
                    output_dir_value,
                    repository_root=root,
                    expected_path=output_dir,
                    label=(
                        "Historical training command --output-dir "
                        f"for {coordinate[0]}/{coordinate[1]}"
                    ),
                )
                checkpoint = summary.get("checkpoint")
                ledger_checkpoint = run.get("checkpoint")
                _require(
                    isinstance(checkpoint, Mapping)
                    and isinstance(ledger_checkpoint, Mapping)
                    and checkpoint == ledger_checkpoint
                    and isinstance(checkpoint.get("path"), str)
                    and _is_sha256(checkpoint.get("sha256"))
                    and type(checkpoint.get("bytes")) is int
                    and cast(int, checkpoint.get("bytes")) > 0,
                    "Historical training checkpoint binding differs between summary and ledger.",
                )
                checkpoint_binding = dict(cast(Mapping[str, Any], checkpoint))
                expected_checkpoint = output_dir / f"{coordinate[0]}-step-1000.pt"
                checkpoint_relative, checkpoint_resolved = (
                    _resolve_historical_command_relative_path(
                        checkpoint_binding["path"],
                        repository_root=root,
                        expected_path=expected_checkpoint,
                        label=(
                            "Historical training checkpoint path "
                            f"for {coordinate[0]}/{coordinate[1]}"
                        ),
                        require_directory=False,
                    )
                )
                checkpoint_opened = _open_secure_regular(
                    checkpoint_resolved,
                    label=f"Historical training checkpoint {coordinate[0]}/{coordinate[1]}",
                )
                try:
                    _require(
                        checkpoint_opened.sha256 == checkpoint_binding["sha256"]
                        and checkpoint_opened.bytes == checkpoint_binding["bytes"],
                        "Historical training checkpoint bytes differ from signed metadata.",
                    )
                    checkpoint_opened.assert_unchanged()
                finally:
                    checkpoint_opened.close()
                rows.append(
                    {
                        "option": "--output-dir",
                        "builder": "training_matrix.build_training_command",
                        "command_sha256": _json_digest(recorded_command),
                        "recorded_command": recorded_command,
                        "recorded_executable": expected_executable,
                        "recorded_relative_path": recorded,
                        "resolved_path": str(resolved),
                        "checkpoint_binding": checkpoint_binding,
                        "checkpoint_relative_path": checkpoint_relative,
                        "resolved_checkpoint_path": str(checkpoint_resolved),
                        "expected_context_profile": context_profiles[coordinate],
                        "scale": coordinate[0],
                        "training_seed": coordinate[1],
                    }
                )
                summary_opened.assert_unchanged()
            finally:
                summary_opened.close()
        _require(
            coordinates == {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS},
            "Historical training command coordinate inventory is incomplete.",
        )
        ledger_opened.assert_unchanged()
        return tuple(sorted(rows, key=lambda row: (row["scale"], row["training_seed"])))
    finally:
        ledger_opened.close()


def _historical_command_path_resolution_claim(
    inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [dict(row) for row in inventory]
    _require(
        len(rows) == len(SCALES) * len(TRAINING_SEEDS),
        "Historical relative command-path inventory is incomplete.",
    )
    output_directory_paths = [
        {
            "scale": row.get("scale"),
            "training_seed": row.get("training_seed"),
            "recorded_relative_path": row.get("recorded_relative_path"),
            "resolved_path": row.get("resolved_path"),
        }
        for row in rows
    ]
    checkpoint_paths = [
        {
            "scale": row.get("scale"),
            "training_seed": row.get("training_seed"),
            "checkpoint_binding": row.get("checkpoint_binding"),
            "recorded_relative_path": row.get("checkpoint_relative_path"),
            "resolved_path": row.get("resolved_checkpoint_path"),
        }
        for row in rows
    ]
    return {
        "schema_version": 1,
        "adapter_scopes": [
            "legacy-training-matrix._validate_training_command-only",
            "legacy-training-matrix._validate_checkpoint-only",
        ],
        "relative_path_fields": [
            {
                "field": "training-summary.command.--output-dir",
                "validated_count": len(output_directory_paths),
                "inventory_sha256": _json_digest(output_directory_paths),
            },
            {
                "field": "training-summary.checkpoint.path",
                "validated_count": len(checkpoint_paths),
                "inventory_sha256": _json_digest(checkpoint_paths),
            },
        ],
        "relative_path_base": "canonical-original-repository-root",
        "source_and_import_validation_cwd": "detached-historical-result-source",
        "cwd_switch": "exact-directory-fd-to-canonical-repository-root",
        "cwd_restoration": "exact-directory-fd-to-detached-result-source",
        "path_values_rewritten": False,
        "parent_traversal_allowed": False,
        "symlink_traversal_allowed": False,
        "validated_relative_path_count": len(rows) * 2,
        "validated_relative_path_inventory_sha256": _json_digest(
            {"schema_version": 1, "paths": rows}
        ),
    }


def _historical_signed_builder_command_inventory(
    repository_root: Path,
    training_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    root = _exact_path(repository_root, label="Historical command repository root", must_exist=True)
    expected_executable = str(root / ARCHIVED_PYTHON_RELATIVE_PATH)
    training_rows = tuple(dict(row) for row in training_inventory)
    _require(
        len(training_rows) == len(SCALES) * len(TRAINING_SEEDS)
        and all(
            row.get("builder") == "training_matrix.build_training_command"
            and row.get("recorded_executable") == expected_executable
            and _is_sha256(row.get("command_sha256"))
            and row.get("command_sha256") == _json_digest(row.get("recorded_command"))
            for row in training_rows
        ),
        "Historical signed training command inventory drifted.",
    )

    def ledger_rows(
        *,
        path: Path,
        expected_sha256: str,
        collection_name: str,
        expected_count: int,
        builder: str,
        coordinate_fields: tuple[str, ...],
    ) -> tuple[dict[str, Any], ...]:
        payload, opened = _load_json_nofollow(path, label=f"Historical {builder} ledger")
        try:
            raw_rows = payload.get(collection_name)
            _require(
                opened.sha256 == expected_sha256
                and isinstance(raw_rows, list)
                and len(raw_rows) == expected_count,
                f"Historical signed {builder} ledger inventory drifted.",
            )
            rows: list[dict[str, Any]] = []
            coordinates: set[tuple[Any, ...]] = set()
            for raw_row in cast(list[Any], raw_rows):
                _require(isinstance(raw_row, Mapping), f"Historical {builder} row is invalid.")
                artifact = cast(Mapping[str, Any], raw_row)
                coordinate = tuple(artifact.get(field) for field in coordinate_fields)
                command = artifact.get("command")
                _require(
                    all(value is not None for value in coordinate)
                    and coordinate not in coordinates
                    and isinstance(command, list)
                    and bool(command)
                    and all(isinstance(item, str) for item in command)
                    and command[0] == expected_executable,
                    f"Historical signed {builder} command or coordinate drifted.",
                )
                coordinates.add(coordinate)
                recorded_command = list(cast(list[str], command))
                rows.append(
                    {
                        "builder": builder,
                        "command_sha256": _json_digest(recorded_command),
                        "coordinate": list(coordinate),
                        "recorded_command": recorded_command,
                        "recorded_executable": expected_executable,
                    }
                )
            opened.assert_unchanged()
            return tuple(sorted(rows, key=lambda row: canonical_json(row["coordinate"])))
        finally:
            opened.close()

    calibration_rows = ledger_rows(
        path=root / HISTORICAL_CALIBRATION_LEDGER,
        expected_sha256=HISTORICAL_CALIBRATION_LEDGER_SHA256,
        collection_name="cells",
        expected_count=len(SCALES) * len(TRAINING_SEEDS),
        builder="calibration_matrix.build_calibration_command",
        coordinate_fields=("scale", "training_seed", "calibration_seed"),
    )
    top_p_rows = ledger_rows(
        path=root / HISTORICAL_TOP_P_LEDGER,
        expected_sha256=HISTORICAL_TOP_P_LEDGER_SHA256,
        collection_name="records",
        expected_count=40,
        builder="top_p_matrix.build_generator_command",
        coordinate_fields=("coordinate_key",),
    )
    return {
        "training_matrix.build_training_command": training_rows,
        "calibration_matrix.build_calibration_command": calibration_rows,
        "top_p_matrix.build_generator_command": top_p_rows,
    }


def _symlink_chain(path: Path) -> tuple[dict[str, Any], ...]:
    current = path
    rows: list[dict[str, Any]] = []
    visited: set[Path] = set()
    for _index in range(32):
        _require(current not in visited, "Archived Python venv alias contains a symlink cycle.")
        visited.add(current)
        before = os.lstat(current)
        if not stat.S_ISLNK(before.st_mode):
            break
        target = os.readlink(current)
        after = os.lstat(current)
        _require(
            _directory_identity(before) == _directory_identity(after),
            "Archived Python venv alias changed while its symlink chain was read.",
        )
        rows.append(
            {
                "path": str(current),
                "target": target,
                "device": before.st_dev,
                "inode": before.st_ino,
                "mode": before.st_mode,
                "uid": before.st_uid,
                "gid": before.st_gid,
            }
        )
        target_path = Path(target)
        current = target_path if target_path.is_absolute() else current.parent / target_path
        current = Path(os.path.abspath(current))
    else:
        raise ValueError("Archived Python venv alias symlink chain is too deep.")
    return tuple(rows)


@dataclass(frozen=True)
class _RetainedInterpreterSpelling:
    path: Path
    target: Path
    file_descriptor: int
    binding: dict[str, Any]


def _interpreter_spelling_binding(
    repository_root: Path,
    runtime_binding: Mapping[str, Any],
    *,
    file_descriptor: int,
) -> dict[str, Any]:
    root = _exact_path(
        repository_root, label="Interpreter spelling repository root", must_exist=True
    )
    spelling = Path(os.path.abspath(root / ARCHIVED_PYTHON_RELATIVE_PATH))
    spelling_parent = _exact_path(
        spelling.parent,
        label="Historical interpreter spelling parent",
        must_exist=True,
    )
    _require(spelling_parent.is_dir(), "Historical interpreter spelling parent is not a directory.")
    runtime_executable = runtime_binding.get("executable")
    runtime_metadata = runtime_binding.get("executable_metadata")
    runtime_sha256 = runtime_binding.get("executable_sha256")
    _require(
        runtime_binding.get("venv_executable") == str(spelling)
        and isinstance(runtime_executable, str)
        and isinstance(runtime_metadata, Mapping)
        and _is_sha256(runtime_sha256),
        "Archived runtime binding cannot authorize the historical interpreter spelling.",
    )
    runtime_executable_path = Path(cast(str, runtime_executable))
    runtime_metadata_map = cast(Mapping[str, Any], runtime_metadata)
    target = spelling.resolve(strict=True)
    proc_target = Path("/proc/self/exe").resolve(strict=True)
    active_target = Path(sys.executable).resolve(strict=True)
    _require(
        target == proc_target == active_target == runtime_executable_path,
        "Historical interpreter spelling does not resolve to the active sealed runtime.",
    )
    opened = os.fstat(file_descriptor)
    current = os.stat(spelling, follow_symlinks=True)
    metadata = _runtime_metadata(target)
    _require(
        _directory_identity(opened) == _directory_identity(current)
        and metadata == dict(runtime_metadata_map)
        and metadata["uid"] == 0
        and metadata["mode"] & (stat.S_IWGRP | stat.S_IWOTH) == 0,
        "Historical interpreter spelling target metadata differs from the sealed runtime.",
    )
    digest = hashlib.sha256()
    offset = 0
    while offset < opened.st_size:
        chunk = os.pread(file_descriptor, min(1 << 20, opened.st_size - offset), offset)
        _require(bool(chunk), "Historical interpreter target read stalled.")
        digest.update(chunk)
        offset += len(chunk)
    _require(
        offset == opened.st_size and digest.hexdigest() == runtime_sha256,
        "Historical interpreter spelling target bytes differ from the sealed runtime.",
    )
    return {
        "schema_version": 1,
        "recorded_executable": str(spelling),
        "alias_symlink_chain": list(_symlink_chain(spelling)),
        "resolved_target": str(target),
        "target_metadata": metadata,
        "target_sha256": runtime_sha256,
    }


@contextmanager
def _retained_interpreter_spelling(
    repository_root: Path,
    runtime_binding: Mapping[str, Any],
) -> Iterator[_RetainedInterpreterSpelling]:
    root = _exact_path(
        repository_root, label="Interpreter spelling repository root", must_exist=True
    )
    spelling = Path(os.path.abspath(root / ARCHIVED_PYTHON_RELATIVE_PATH))
    spelling_parent = _exact_path(
        spelling.parent,
        label="Historical interpreter spelling parent",
        must_exist=True,
    )
    _require(spelling_parent.is_dir(), "Historical interpreter spelling parent is not a directory.")
    descriptor = os.open(spelling, os.O_RDONLY | os.O_CLOEXEC)
    try:
        binding = _interpreter_spelling_binding(
            root,
            runtime_binding,
            file_descriptor=descriptor,
        )
        retained = _RetainedInterpreterSpelling(
            path=spelling,
            target=Path(cast(str, binding["resolved_target"])),
            file_descriptor=descriptor,
            binding=binding,
        )
        try:
            yield retained
        finally:
            _require(
                _interpreter_spelling_binding(
                    root,
                    runtime_binding,
                    file_descriptor=descriptor,
                )
                == binding,
                "Historical interpreter spelling alias or retained target drifted.",
            )
    finally:
        os.close(descriptor)


def _historical_builder_adapter_claim(
    retained: _RetainedInterpreterSpelling,
    inventories: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    observed_invocations: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    expected_counts = {
        "training_matrix.build_training_command": 10,
        "calibration_matrix.build_calibration_command": 10,
        "top_p_matrix.build_generator_command": 40,
    }
    expected_coordinate_counts = {
        (scale, seed): (71 if (scale, seed) == (SCALES[0], TRAINING_SEEDS[0]) else 69)
        for scale in SCALES
        for seed in TRAINING_SEEDS
    }
    builders: list[dict[str, Any]] = []
    total_invocations = 0
    for name in sorted(expected_counts):
        rows = [dict(row) for row in inventories.get(name, ())]
        _require(len(rows) == expected_counts[name], f"Historical {name} inventory is incomplete.")
        expected_invocations: dict[str, int] = {}
        for row in rows:
            digest = row.get("command_sha256")
            _require(_is_sha256(digest), f"Historical {name} command digest is invalid.")
            count = 1
            if name == "training_matrix.build_training_command":
                coordinate = (row.get("scale"), row.get("training_seed"))
                _require(
                    coordinate in expected_coordinate_counts,
                    "Historical training builder coordinate is invalid.",
                )
                count = expected_coordinate_counts[cast(tuple[str, int], coordinate)]
            expected_invocations[cast(str, digest)] = count
        if observed_invocations is not None:
            _require(
                dict(observed_invocations.get(name, {})) == expected_invocations,
                f"Historical {name} invocation multiplicities drifted.",
            )
        invocation_rows = [
            {"command_sha256": digest, "invocations": count}
            for digest, count in sorted(expected_invocations.items())
        ]
        invocation_count = sum(expected_invocations.values())
        total_invocations += invocation_count
        builders.append(
            {
                "builder": name,
                "invocation_entry_cwd": (
                    "canonical-repository-root"
                    if name == "training_matrix.build_training_command"
                    else "detached-result-source"
                ),
                "original_call_cwd": (
                    "canonical-repository-root"
                    if name
                    in {
                        "calibration_matrix.build_calibration_command",
                        "training_matrix.build_training_command",
                    }
                    else "detached-result-source"
                ),
                "invocation_exit_cwd": (
                    "canonical-repository-root"
                    if name == "training_matrix.build_training_command"
                    else "detached-result-source"
                ),
                "signed_command_count": len(rows),
                "signed_command_inventory_sha256": _json_digest(
                    {"schema_version": 1, "commands": rows}
                ),
                "expected_invocation_count": invocation_count,
                "expected_invocation_multiset_sha256": _json_digest(
                    {"schema_version": 1, "invocations": invocation_rows}
                ),
            }
        )
    return {
        "schema_version": 1,
        "actual_child_process_executable_unchanged": True,
        "sys_executable_mutated": False,
        "reexec_performed": False,
        "adapter_operation": "clone-original-builder-result-and-replace-index-zero-only",
        "nonzero_argv_bytes_and_order_preserved": True,
        "builder_replay_policy": "exact-signed-membership-complete-coverage-repeats-allowed",
        "adapter_context_install_and_restore_cwd": "detached-result-source",
        "callable_identity_restored": True,
        "interpreter_spelling": dict(retained.binding),
        "builders": builders,
        "signed_command_count": sum(expected_counts.values()),
        "expected_builder_invocation_count": total_invocations,
    }


@contextmanager
def _legacy_builder_spelling_adapters(
    specs: Sequence[tuple[Any, str, str]],
    *,
    repository_root: Path,
    detached_root: Path,
    retained: _RetainedInterpreterSpelling,
    inventories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Iterator[dict[str, dict[str, int]]]:
    _require(
        threading.current_thread() is threading.main_thread() and threading.active_count() == 1,
        "Legacy builder adapters require an isolated single-threaded archived child.",
    )
    originals: list[tuple[Any, str, Callable[..., Any], Callable[..., Any]]] = []
    validated: dict[str, dict[str, int]] = {}
    spec_keys = [(id(module), attribute) for module, attribute, _claim_name in specs]
    claim_names = [claim_name for _module, _attribute, claim_name in specs]
    expected_claim_names = {
        "training_matrix.build_training_command",
        "calibration_matrix.build_calibration_command",
        "top_p_matrix.build_generator_command",
    }
    _require(
        len(specs) == 3
        and len(set(spec_keys)) == 3
        and len(set(claim_names)) == 3
        and set(claim_names) == expected_claim_names
        and set(inventories) == expected_claim_names,
        "Legacy builder adapter specification is not the exact closed three-builder scope.",
    )
    try:
        for module, attribute, claim_name in specs:
            raw_original = getattr(module, attribute, None)
            _require(
                callable(raw_original)
                and getattr(raw_original, "__name__", None) == attribute
                and not bool(getattr(raw_original, "_adaptive_v4_archived_builder_adapter", False)),
                f"Legacy builder {claim_name} is unavailable, aliased, or already adapted.",
            )
            original = cast(Callable[..., Any], raw_original)
            rows = inventories.get(claim_name, ())
            expected_commands: dict[str, list[str]] = {}
            for row in rows:
                digest = row.get("command_sha256")
                command = row.get("recorded_command")
                _require(
                    _is_sha256(digest)
                    and isinstance(command, list)
                    and all(isinstance(item, str) for item in command)
                    and command[0] == str(retained.path)
                    and digest == _json_digest(command)
                    and cast(str, digest) not in expected_commands,
                    f"Legacy builder {claim_name} signed command inventory is invalid.",
                )
                expected_commands[cast(str, digest)] = list(cast(list[str], command))
            _require(
                bool(expected_commands), f"Legacy builder {claim_name} has no signed commands."
            )
            seen: dict[str, int] = {}
            validated[claim_name] = seen

            def adapter(
                *arguments: Any,
                _original: Callable[..., Any] = original,
                _claim_name: str = claim_name,
                _expected_commands: Mapping[str, list[str]] = expected_commands,
                _seen: dict[str, int] = seen,
                **keywords: Any,
            ) -> list[str]:
                _require(
                    threading.current_thread() is threading.main_thread()
                    and threading.active_count() == 1
                    and sys.executable == str(retained.target),
                    f"Legacy builder {_claim_name} runtime or thread scope drifted.",
                )
                expected_entry_mode = (
                    "canonical-root"
                    if _claim_name == "training_matrix.build_training_command"
                    else "detached"
                )
                _require(
                    _historical_cwd_authority_mode(
                        repository_root=repository_root,
                        detached_root=detached_root,
                    )
                    == expected_entry_mode,
                    f"Legacy builder {_claim_name} did not enter from its exact invocation cwd.",
                )
                before_binding = _interpreter_spelling_binding(
                    retained.path.parents[2],
                    {
                        "executable": str(retained.target),
                        "venv_executable": str(retained.path),
                        "executable_metadata": retained.binding["target_metadata"],
                        "executable_sha256": retained.binding["target_sha256"],
                    },
                    file_descriptor=retained.file_descriptor,
                )
                try:
                    if _claim_name == "calibration_matrix.build_calibration_command":
                        with _scoped_historical_command_resolution_cwd(
                            repository_root=repository_root,
                            detached_root=detached_root,
                        ):
                            original_result = _original(*arguments, **keywords)
                    else:
                        original_result = _original(*arguments, **keywords)
                finally:
                    after_binding = _interpreter_spelling_binding(
                        retained.path.parents[2],
                        {
                            "executable": str(retained.target),
                            "venv_executable": str(retained.path),
                            "executable_metadata": retained.binding["target_metadata"],
                            "executable_sha256": retained.binding["target_sha256"],
                        },
                        file_descriptor=retained.file_descriptor,
                    )
                    _require(
                        after_binding == before_binding
                        and sys.executable == str(retained.target)
                        and threading.current_thread() is threading.main_thread()
                        and threading.active_count() == 1,
                        f"Legacy builder {_claim_name} runtime drifted during reconstruction.",
                    )
                    _require(
                        _historical_cwd_authority_mode(
                            repository_root=repository_root,
                            detached_root=detached_root,
                        )
                        == expected_entry_mode,
                        f"Legacy builder {_claim_name} did not restore its exact invocation cwd.",
                    )
                _require(
                    isinstance(original_result, list)
                    and bool(original_result)
                    and all(isinstance(item, str) for item in original_result)
                    and original_result[0] == str(retained.target),
                    f"Legacy builder {_claim_name} did not return the active resolved interpreter.",
                )
                original_snapshot = list(cast(list[str], original_result))
                adapted = list(original_snapshot)
                adapted[0] = str(retained.path)
                digest = _json_digest(adapted)
                _require(
                    digest in _expected_commands
                    and adapted == _expected_commands[digest]
                    and original_result == original_snapshot
                    and adapted[1:] == original_snapshot[1:],
                    f"Legacy builder {_claim_name} output differs from signed command evidence.",
                )
                _seen[digest] = _seen.get(digest, 0) + 1
                return adapted

            adapter._adaptive_v4_archived_builder_adapter = True  # type: ignore[attr-defined]
            setattr(module, attribute, adapter)
            originals.append((module, attribute, original, adapter))
        yield validated
    finally:
        identities_held = True
        for module, attribute, original, adapter in reversed(originals):
            identities_held = identities_held and getattr(module, attribute, None) is adapter
            setattr(module, attribute, original)
            identities_held = identities_held and getattr(module, attribute, None) is original
        _require(
            identities_held
            and threading.current_thread() is threading.main_thread()
            and threading.active_count() == 1
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "detached",
            "Legacy builder adapter callable identity or thread scope drifted.",
        )


def _directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
    )


def _cwd_matches(path: Path, identity: tuple[int, int, int, int, int]) -> bool:
    try:
        return (
            Path.cwd() == path
            and _directory_identity(os.stat(".", follow_symlinks=False)) == identity
        )
    except OSError:
        return False


def _native_task_ids() -> tuple[int, ...]:
    try:
        raw_entries = os.listdir("/proc/self/task")
    except OSError as error:
        raise ValueError("Archived child native task inventory is unavailable.") from error
    _require(
        bool(raw_entries) and all(entry.isascii() and entry.isdecimal() for entry in raw_entries),
        "Archived child native task inventory is malformed.",
    )
    task_ids = tuple(sorted(int(entry) for entry in raw_entries))
    _require(
        len(task_ids) == len(set(task_ids)) and threading.get_native_id() in task_ids,
        "Archived child native task inventory is inconsistent.",
    )
    return task_ids


def _native_task_cwd(
    task_id: int,
) -> tuple[str, tuple[int, int, int, int, int]]:
    cwd_link = Path("/proc/self/task") / str(task_id) / "cwd"
    try:
        path = os.readlink(cwd_link)
        metadata = os.stat(cwd_link)
    except OSError as error:
        raise ValueError("Archived child native task cwd is unavailable.") from error
    _require(
        os.path.isabs(path) and " (deleted)" not in path and stat.S_ISDIR(metadata.st_mode),
        "Archived child native task cwd is unsafe.",
    )
    return path, _directory_identity(metadata)


def _native_task_cwd_inventory(
    task_ids: Sequence[int],
) -> tuple[tuple[int, str, tuple[int, int, int, int, int]], ...]:
    _require(
        tuple(task_ids) == _native_task_ids(),
        "Archived child native task set drifted during cwd observation.",
    )
    return tuple((task_id, *_native_task_cwd(task_id)) for task_id in task_ids)


def _require_native_task_cwds(
    inventory: Sequence[tuple[int, str, tuple[int, int, int, int, int]]],
    *,
    expected_path: Path,
    expected_identity: tuple[int, int, int, int, int],
    label: str,
) -> None:
    _require(
        bool(inventory)
        and all(
            path == str(expected_path) and identity == expected_identity
            for _task_id, path, identity in inventory
        ),
        f"{label} native task cwd inventory drifted.",
    )


def _await_exact_native_task_cwd_inventory(
    expected_task_ids: tuple[int, ...],
    *,
    expected_path: Path,
    expected_identity: tuple[int, int, int, int, int],
    label: str,
) -> tuple[tuple[int, str, tuple[int, int, int, int, int]], ...]:
    deadline = time.monotonic() + HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS
    last_error: BaseException | None = None
    while True:
        try:
            _require(
                _native_task_ids() == expected_task_ids,
                f"{label} native task set has not stabilized.",
            )
            inventory = _native_task_cwd_inventory(expected_task_ids)
            _require_native_task_cwds(
                inventory,
                expected_path=expected_path,
                expected_identity=expected_identity,
                label=label,
            )
            return inventory
        except (OSError, ValueError) as error:
            last_error = error
        if time.monotonic() >= deadline:
            raise ValueError(f"{label} native task inventory did not stabilize.") from last_error
        time.sleep(0.001)


def _historical_thread_fs_isolation_static_policy() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "platform": "linux-procfs-cpython",
        "api": "os.unshare",
        "clone_flag_name": "CLONE_FS",
        "clone_flag_value": 0x200,
        "unshare_call_count": 1,
        "ordering": [
            "capture-detached-native-task-baseline",
            "start-pre-unshare-python-probe-thread",
            "unshare-main-thread-fs-context",
            "move-main-only-to-canonical-root",
            "observe-probe-and-preexisting-non-main-tasks-at-detached-root",
            "restore-main-to-detached-root",
            "join-probe-thread-and-freeze-stable-native-task-baseline",
            "enter-capability-gated-cwd-scopes",
        ],
        "probe_semantics": (
            "pre-unshare-python-thread-shares-original-fs-struct-and-remains-detached-"
            "while-unshared-main-enters-canonical-root"
        ),
        "capability_semantics": (
            "private-process-local-object-identity-required-at-entry-during-and-finally-"
            "for-every-cwd-scope"
        ),
        "native_task_boundary": (
            "exact-post-probe-tid-set-and-cwd-identities-at-every-scope-boundary;"
            "new-or-exited-native-tasks-are-unsupported-and-fail-closed"
        ),
        "dynamic_observation_verification": (
            "signed-canonical-semantic-observation-digest-parent-recomputed;"
            "raw-task-and-directory-identities-private-live-capability-only"
        ),
    }


def _historical_thread_fs_semantic_observation_sha256(
    observation: Mapping[str, Any],
) -> str:
    _require(
        set(observation) == set(_HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_FIELDS),
        "Historical thread fs-isolation semantic observation schema drifted.",
    )
    projected = {
        field: observation[field] for field in _HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_FIELDS
    }
    return _json_digest(
        {
            "domain": HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_DOMAIN,
            "schema_version": 1,
            "observation": projected,
        }
    )


def _verify_historical_thread_fs_isolation_claim(raw: object) -> None:
    expected_policy = _historical_thread_fs_isolation_static_policy()
    _require(
        isinstance(raw, Mapping)
        and set(raw) == {"schema_version", "static_policy", "dynamic_observation"}
        and type(raw.get("schema_version")) is int
        and raw.get("schema_version") == 2
        and isinstance(raw.get("static_policy"), Mapping)
        and canonical_json(raw.get("static_policy")) == canonical_json(expected_policy),
        "Historical thread fs-isolation static claim drifted.",
    )
    dynamic = cast(Mapping[str, Any], raw).get("dynamic_observation")
    _require(
        isinstance(dynamic, Mapping)
        and set(dynamic)
        == {
            "python_thread_count_before_probe",
            "python_thread_count_during_probe",
            "python_thread_count_after_probe",
            "native_task_count_before_probe",
            "native_task_count_during_probe",
            "native_task_count_after_probe",
            "preexisting_non_main_native_task_count",
            "probe_thread_native_task_count",
            "main_transition",
            "probe_thread_transition",
            "preexisting_non_main_transition",
            "task_set_restored_after_probe",
            "all_tasks_detached_after_probe",
            "semantic_observation_sha256",
        },
        "Historical thread fs-isolation dynamic claim schema drifted.",
    )
    dynamic_map = cast(Mapping[str, Any], dynamic)
    numeric_fields = (
        "python_thread_count_before_probe",
        "python_thread_count_during_probe",
        "python_thread_count_after_probe",
        "native_task_count_before_probe",
        "native_task_count_during_probe",
        "native_task_count_after_probe",
        "preexisting_non_main_native_task_count",
        "probe_thread_native_task_count",
    )
    before = dynamic_map.get("native_task_count_before_probe")
    during = dynamic_map.get("native_task_count_during_probe")
    after = dynamic_map.get("native_task_count_after_probe")
    main_transition = dynamic_map.get("main_transition")
    probe_transition = dynamic_map.get("probe_thread_transition")
    preexisting_transition = dynamic_map.get("preexisting_non_main_transition")
    semantic_observation = {
        field: dynamic_map[field] for field in _HISTORICAL_THREAD_FS_SEMANTIC_OBSERVATION_FIELDS
    }
    _require(
        all(type(dynamic_map.get(field)) is int for field in numeric_fields)
        and type(before) is int
        and before >= 1
        and during == before + 1
        and after == before
        and dynamic_map.get("python_thread_count_before_probe") == 1
        and dynamic_map.get("python_thread_count_during_probe") == 2
        and dynamic_map.get("python_thread_count_after_probe") == 1
        and dynamic_map.get("preexisting_non_main_native_task_count") == before - 1
        and dynamic_map.get("probe_thread_native_task_count") == 1
        and type(main_transition) is list
        and all(type(item) is str for item in main_transition)
        and main_transition == ["detached", "canonical-root", "detached"]
        and type(probe_transition) is list
        and all(type(item) is str for item in probe_transition)
        and probe_transition == ["detached", "detached", "detached"]
        and type(preexisting_transition) is list
        and all(type(item) is str for item in preexisting_transition)
        and preexisting_transition == ["detached", "detached", "detached"]
        and dynamic_map.get("task_set_restored_after_probe") is True
        and dynamic_map.get("all_tasks_detached_after_probe") is True
        and type(dynamic_map.get("semantic_observation_sha256")) is str
        and _is_sha256(dynamic_map.get("semantic_observation_sha256"))
        and dynamic_map.get("semantic_observation_sha256")
        == _historical_thread_fs_semantic_observation_sha256(semantic_observation),
        "Historical thread fs-isolation dynamic observation drifted.",
    )


def _assert_thread_fs_isolation_capability(
    *,
    repository_root: Path,
    detached_root: Path,
    main_location: str,
) -> _HistoricalThreadFsIsolationCapability:
    capability = _HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY
    _require(
        _HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED
        and type(capability) is _HistoricalThreadFsIsolationCapability
        and capability.seal is _HISTORICAL_THREAD_FS_ISOLATION_SEAL
        and _historical_thread_fs_capability_is_installed(capability)
        and _get_installed_historical_thread_fs_capability() is capability,
        "Historical cwd scope lacks the private thread fs-isolation capability.",
    )
    exact_capability = cast(_HistoricalThreadFsIsolationCapability, capability)
    try:
        claim = json.loads(exact_capability.claim_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Historical cwd capability claim bytes are invalid.") from error
    _require(
        bool(exact_capability.claim_bytes)
        and exact_capability.claim_sha256
        == hashlib.sha256(exact_capability.claim_bytes).hexdigest()
        and isinstance(claim, Mapping)
        and canonical_json(claim) == exact_capability.claim_bytes,
        "Historical cwd capability claim is not canonical.",
    )
    _verify_historical_thread_fs_isolation_claim(claim)
    _require(
        main_location in {"detached", "canonical-root"}
        and repository_root == exact_capability.repository_root
        and detached_root == exact_capability.detached_root
        and threading.current_thread() is threading.main_thread()
        and threading.get_native_id() == exact_capability.main_native_id
        and threading.active_count() == 1
        and _native_task_ids() == exact_capability.baseline_task_ids
        and _directory_identity(os.stat(repository_root, follow_symlinks=False))
        == exact_capability.root_identity
        and _directory_identity(os.stat(detached_root, follow_symlinks=False))
        == exact_capability.detached_identity,
        "Historical cwd scope thread, native task set, or directory authority drifted.",
    )
    expected_main_path = repository_root if main_location == "canonical-root" else detached_root
    expected_main_identity = (
        exact_capability.root_identity
        if main_location == "canonical-root"
        else exact_capability.detached_identity
    )
    current_inventory = _native_task_cwd_inventory(exact_capability.baseline_task_ids)
    baseline_by_task = {
        task_id: (path, identity) for task_id, path, identity in exact_capability.baseline_task_cwds
    }
    _require(
        _cwd_matches(expected_main_path, expected_main_identity)
        and baseline_by_task.get(exact_capability.main_native_id)
        == (str(detached_root), exact_capability.detached_identity)
        and all(
            (
                path == str(expected_main_path) and identity == expected_main_identity
                if task_id == exact_capability.main_native_id
                else baseline_by_task.get(task_id)
                == (path, identity)
                == (str(detached_root), exact_capability.detached_identity)
            )
            for task_id, path, identity in current_inventory
        ),
        "Historical cwd scope per-task cwd authority drifted.",
    )
    return exact_capability


def _unshare_validating_thread_fs_context(
    *,
    repository_root: Path,
    detached_root: Path,
) -> dict[str, Any]:
    global _HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY
    global _HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED

    _require(
        not _HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED
        and _HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY is None,
        "Historical thread fs-isolation may be established exactly once.",
    )
    _HISTORICAL_THREAD_FS_UNSHARE_ATTEMPTED = True
    root = _exact_path(
        repository_root,
        label="Historical thread fs-isolation repository root",
        must_exist=True,
    )
    detached = _exact_path(
        detached_root,
        label="Historical thread fs-isolation detached root",
        must_exist=True,
    )
    raw_unshare = getattr(os, "unshare", None)
    clone_fs = getattr(os, "CLONE_FS", None)
    _require(
        sys.platform == "linux"
        and sys.implementation.name == "cpython"
        and Path("/proc/self/task").is_dir()
        and callable(raw_unshare)
        and type(clone_fs) is int
        and clone_fs == 0x200
        and root.is_dir()
        and detached.is_dir()
        and root != detached
        and threading.current_thread() is threading.main_thread()
        and threading.active_count() == 1,
        "Historical thread fs-isolation platform or Python scope is unsupported.",
    )
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Historical thread fs-isolation requires O_NOFOLLOW.")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | cast(int, no_follow)
    detached_fd = os.open(detached, flags)
    root_fd = -1
    probe_thread: threading.Thread | None = None
    probe_ready = threading.Event()
    root_probe_requested = threading.Event()
    root_probe_observed = threading.Event()
    restoration_requested = threading.Event()
    probe_finished = threading.Event()
    probe_errors: list[BaseException] = []
    probe_observations: list[dict[str, Any]] = []
    probe_control = {"abort": False}
    failure: BaseException | None = None
    unshare_succeeded = False
    main_entered_root = False
    main_restored = False
    initial_task_ids: tuple[int, ...] = ()
    initial_inventory: tuple[tuple[int, str, tuple[int, int, int, int, int]], ...] = ()
    during_task_ids: tuple[int, ...] = ()
    during_inventory: tuple[tuple[int, str, tuple[int, int, int, int, int]], ...] = ()
    root_phase_inventory: tuple[tuple[int, str, tuple[int, int, int, int, int]], ...] = ()
    restored_phase_inventory: tuple[tuple[int, str, tuple[int, int, int, int, int]], ...] = ()
    post_task_ids: tuple[int, ...] = ()
    post_inventory: tuple[tuple[int, str, tuple[int, int, int, int, int]], ...] = ()
    main_native_id = threading.get_native_id()
    try:
        root_fd = os.open(root, flags)
        detached_identity = _directory_identity(os.fstat(detached_fd))
        root_identity = _directory_identity(os.fstat(root_fd))
        _require(
            detached_identity == _directory_identity(os.stat(detached, follow_symlinks=False))
            and root_identity == _directory_identity(os.stat(root, follow_symlinks=False))
            and stat.S_ISDIR(detached_identity[2])
            and stat.S_ISDIR(root_identity[2])
            and _cwd_matches(detached, detached_identity),
            "Historical thread fs-isolation directory identity is unsafe.",
        )
        initial_task_ids = _native_task_ids()
        initial_inventory = _native_task_cwd_inventory(initial_task_ids)
        _require_native_task_cwds(
            initial_inventory,
            expected_path=detached,
            expected_identity=detached_identity,
            label="Pre-probe detached",
        )

        def probe() -> None:
            try:
                probe_native_id = threading.get_native_id()
                before = _native_task_cwd(probe_native_id)
                _require(
                    _cwd_matches(detached, detached_identity)
                    and before == (str(detached), detached_identity),
                    "Pre-unshare Python probe did not inherit the detached fs context.",
                )
                probe_observations.append(
                    {"phase": "before-unshare", "task_id": probe_native_id, "cwd": before}
                )
                probe_ready.set()
                _require(
                    root_probe_requested.wait(timeout=HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS),
                    "Pre-unshare Python probe timed out waiting for the root phase.",
                )
                if probe_control["abort"]:
                    return
                while_root = _native_task_cwd(probe_native_id)
                _require(
                    _cwd_matches(detached, detached_identity)
                    and while_root == (str(detached), detached_identity),
                    "Pre-unshare Python probe followed the main task into the canonical root.",
                )
                probe_observations.append(
                    {"phase": "main-at-root", "task_id": probe_native_id, "cwd": while_root}
                )
                root_probe_observed.set()
                _require(
                    restoration_requested.wait(timeout=HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS),
                    "Pre-unshare Python probe timed out waiting for main restoration.",
                )
                after = _native_task_cwd(probe_native_id)
                _require(
                    _cwd_matches(detached, detached_identity)
                    and after == (str(detached), detached_identity),
                    "Pre-unshare Python probe drifted after main restoration.",
                )
                probe_observations.append(
                    {"phase": "main-restored", "task_id": probe_native_id, "cwd": after}
                )
            except BaseException as error:
                probe_errors.append(error)
            finally:
                probe_ready.set()
                root_probe_observed.set()
                probe_finished.set()

        probe_thread = threading.Thread(
            target=probe,
            name="adaptive-v4-pre-unshare-fs-probe",
            daemon=False,
        )
        probe_thread.start()
        _require(
            probe_ready.wait(timeout=HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS)
            and not probe_errors
            and threading.active_count() == 2,
            "Pre-unshare Python probe failed to become ready.",
        )
        probe_native_id = cast(int, probe_thread.native_id)
        during_task_ids = _native_task_ids()
        _require(
            set(during_task_ids) == set(initial_task_ids) | {probe_native_id}
            and len(during_task_ids) == len(initial_task_ids) + 1,
            "Pre-unshare Python probe native task inventory drifted.",
        )
        during_inventory = _native_task_cwd_inventory(during_task_ids)
        _require_native_task_cwds(
            during_inventory,
            expected_path=detached,
            expected_identity=detached_identity,
            label="Probe-ready detached",
        )
        cast(Callable[[int], None], raw_unshare)(cast(int, clone_fs))
        unshare_succeeded = True
        os.fchdir(root_fd)
        main_entered_root = True
        root_phase_inventory = _native_task_cwd_inventory(during_task_ids)
        _require(
            _cwd_matches(root, root_identity)
            and all(
                (
                    path == str(root) and identity == root_identity
                    if task_id == main_native_id
                    else path == str(detached) and identity == detached_identity
                )
                for task_id, path, identity in root_phase_inventory
            ),
            "CLONE_FS probe did not isolate the main native task cwd.",
        )
        root_probe_requested.set()
        _require(
            root_probe_observed.wait(timeout=HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS)
            and not probe_errors,
            "Pre-unshare Python probe failed during the canonical-root phase.",
        )
    except BaseException as error:
        failure = error
    finally:
        if unshare_succeeded:
            try:
                os.fchdir(detached_fd)
                main_restored = _cwd_matches(detached, detached_identity)
                if during_task_ids:
                    restored_phase_inventory = _native_task_cwd_inventory(during_task_ids)
                    _require_native_task_cwds(
                        restored_phase_inventory,
                        expected_path=detached,
                        expected_identity=detached_identity,
                        label="Restored detached",
                    )
            except BaseException as error:
                main_restored = False
                if failure is None:
                    failure = error
        else:
            try:
                main_restored = _cwd_matches(detached, _directory_identity(os.fstat(detached_fd)))
            except BaseException as error:
                main_restored = False
                if failure is None:
                    failure = error
        if not main_entered_root:
            probe_control["abort"] = True
        for event in (root_probe_requested, restoration_requested):
            try:
                event.set()
            except BaseException as error:
                if failure is None:
                    failure = error
        if probe_thread is not None:
            try:
                probe_thread.join(timeout=HISTORICAL_THREAD_FS_PROBE_TIMEOUT_SECONDS)
                if probe_thread.is_alive() and failure is None:
                    failure = ValueError("Pre-unshare Python probe did not terminate.")
            except BaseException as error:
                if failure is None:
                    failure = error
        if root_fd >= 0:
            try:
                os.close(root_fd)
            except BaseException as error:
                if failure is None:
                    failure = error
        try:
            os.close(detached_fd)
        except BaseException as error:
            if failure is None:
                failure = error

    if failure is not None:
        raise ValueError("Historical thread fs-isolation probe failed.") from failure
    _require(
        unshare_succeeded
        and main_entered_root
        and main_restored
        and probe_finished.is_set()
        and not probe_errors
        and len(probe_observations) == 3
        and threading.active_count() == 1,
        "Historical thread fs-isolation probe did not complete exactly.",
    )
    post_inventory = _await_exact_native_task_cwd_inventory(
        initial_task_ids,
        expected_path=detached,
        expected_identity=detached_identity,
        label="Post-probe detached",
    )
    post_task_ids = tuple(task_id for task_id, _path, _identity in post_inventory)
    semantic_observation = {
        "python_thread_count_before_probe": 1,
        "python_thread_count_during_probe": 2,
        "python_thread_count_after_probe": 1,
        "native_task_count_before_probe": len(initial_task_ids),
        "native_task_count_during_probe": len(during_task_ids),
        "native_task_count_after_probe": len(post_task_ids),
        "preexisting_non_main_native_task_count": len(initial_task_ids) - 1,
        "probe_thread_native_task_count": 1,
        "main_transition": ["detached", "canonical-root", "detached"],
        "probe_thread_transition": ["detached", "detached", "detached"],
        "preexisting_non_main_transition": ["detached", "detached", "detached"],
        "task_set_restored_after_probe": True,
        "all_tasks_detached_after_probe": True,
    }
    claim = {
        "schema_version": 2,
        "static_policy": _historical_thread_fs_isolation_static_policy(),
        "dynamic_observation": {
            **semantic_observation,
            "semantic_observation_sha256": (
                _historical_thread_fs_semantic_observation_sha256(semantic_observation)
            ),
        },
    }
    _verify_historical_thread_fs_isolation_claim(claim)
    claim_bytes = canonical_json(claim)
    capability = _HistoricalThreadFsIsolationCapability(
        seal=_HISTORICAL_THREAD_FS_ISOLATION_SEAL,
        repository_root=root,
        detached_root=detached,
        root_identity=root_identity,
        detached_identity=detached_identity,
        main_native_id=main_native_id,
        baseline_task_ids=post_task_ids,
        baseline_task_cwds=post_inventory,
        claim_bytes=claim_bytes,
        claim_sha256=hashlib.sha256(claim_bytes).hexdigest(),
    )
    _install_historical_thread_fs_capability(capability)
    _HISTORICAL_THREAD_FS_ISOLATION_CAPABILITY = capability
    _assert_thread_fs_isolation_capability(
        repository_root=root,
        detached_root=detached,
        main_location="detached",
    )
    return cast(dict[str, Any], json.loads(claim_bytes))


@contextmanager
def _scoped_historical_command_resolution_cwd(
    *,
    repository_root: Path,
    detached_root: Path,
) -> Iterator[None]:
    root = _exact_path(repository_root, label="Historical command repository root", must_exist=True)
    detached = _exact_path(
        detached_root,
        label="Detached historical result-source root",
        must_exist=True,
    )
    capability = _assert_thread_fs_isolation_capability(
        repository_root=root,
        detached_root=detached,
        main_location=(
            "canonical-root" if _HISTORICAL_ROOT_CWD_AUTHORITY.get() is not None else "detached"
        ),
    )
    active_authority = _HISTORICAL_ROOT_CWD_AUTHORITY.get()
    if active_authority is not None:
        (
            active_capability,
            active_root,
            active_detached,
            root_identity,
            detached_identity,
        ) = active_authority
        _require(
            capability is active_capability
            and root == active_root
            and detached == active_detached
            and _cwd_matches(root, root_identity)
            and _directory_identity(os.stat(detached, follow_symlinks=False)) == detached_identity
            and threading.current_thread() is threading.main_thread()
            and threading.active_count() == 1,
            "Nested historical command cwd scope lacks exact outer root authority.",
        )
        try:
            yield
        finally:
            _assert_thread_fs_isolation_capability(
                repository_root=root,
                detached_root=detached,
                main_location="canonical-root",
            )
            _require(
                _HISTORICAL_ROOT_CWD_AUTHORITY.get() is active_authority
                and _cwd_matches(root, root_identity)
                and _directory_identity(os.stat(detached, follow_symlinks=False))
                == detached_identity
                and threading.current_thread() is threading.main_thread()
                and threading.active_count() == 1,
                "Nested historical command cwd scope drifted from outer authority.",
            )
        return
    _require(
        root.is_dir()
        and detached.is_dir()
        and root != detached
        and Path.cwd() == detached
        and active_authority is None
        and threading.current_thread() is threading.main_thread()
        and threading.active_count() == 1,
        "Historical command cwd scope did not start in the exact detached result source.",
    )
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Historical command cwd scope requires O_NOFOLLOW support.")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | cast(int, no_follow)
    detached_fd = os.open(".", flags)
    root_fd = -1
    try:
        root_fd = os.open(root, flags)
        detached_identity = _directory_identity(os.fstat(detached_fd))
        root_identity = _directory_identity(os.fstat(root_fd))
        _require(
            detached_identity == _directory_identity(os.stat(detached, follow_symlinks=False))
            and root_identity == _directory_identity(os.stat(root, follow_symlinks=False))
            and stat.S_ISDIR(detached_identity[2])
            and stat.S_ISDIR(root_identity[2]),
            "Historical command cwd directory identity is unsafe.",
        )
        authority_token: Any | None = None
        authority_value = (capability, root, detached, root_identity, detached_identity)
        entered_root = False
        cleanup_errors: list[BaseException] = []
        try:
            os.fchdir(root_fd)
            entered_root = True
            authority_token = _HISTORICAL_ROOT_CWD_AUTHORITY.set(authority_value)
            _assert_thread_fs_isolation_capability(
                repository_root=root,
                detached_root=detached,
                main_location="canonical-root",
            )
            _require(
                _cwd_matches(root, root_identity),
                "Historical command cwd did not enter the exact canonical repository root.",
            )
            yield
        finally:
            canonical_root_held = _cwd_matches(root, root_identity)
            authority_held = False
            try:
                authority_held = _HISTORICAL_ROOT_CWD_AUTHORITY.get() is authority_value
            except BaseException as error:
                cleanup_errors.append(error)
            if entered_root:
                try:
                    _assert_thread_fs_isolation_capability(
                        repository_root=root,
                        detached_root=detached,
                        main_location="canonical-root",
                    )
                except BaseException as error:
                    cleanup_errors.append(error)
            if authority_token is not None:
                try:
                    _HISTORICAL_ROOT_CWD_AUTHORITY.reset(authority_token)
                except BaseException as error:
                    cleanup_errors.append(error)
            elif entered_root:
                cleanup_errors.append(
                    ValueError("Historical command cwd authority was not established.")
                )
            if _HISTORICAL_ROOT_CWD_AUTHORITY.get() is not None:
                try:
                    _HISTORICAL_ROOT_CWD_AUTHORITY.set(None)
                except BaseException as error:
                    cleanup_errors.append(error)
            authority_cleared = _HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
            try:
                os.fchdir(detached_fd)
            except BaseException as error:
                cleanup_errors.append(error)
            restored = _cwd_matches(detached, detached_identity)
            if restored:
                try:
                    _assert_thread_fs_isolation_capability(
                        repository_root=root,
                        detached_root=detached,
                        main_location="detached",
                    )
                except BaseException as error:
                    cleanup_errors.append(error)
            if not (
                entered_root
                and canonical_root_held
                and authority_held
                and authority_cleared
                and restored
            ):
                cleanup_errors.append(
                    ValueError(
                        "Historical command cwd drifted or failed exact detached-root restoration."
                    )
                )
            if cleanup_errors:
                raise ValueError(
                    "Historical command cwd cleanup failed closed after exact restoration attempts."
                ) from cleanup_errors[0]
    finally:
        if root_fd >= 0:
            os.close(root_fd)
        os.close(detached_fd)


@contextmanager
def _scoped_historical_source_validation_cwd(
    *,
    repository_root: Path,
    detached_root: Path,
) -> Iterator[None]:
    root = _exact_path(
        repository_root,
        label="Historical source-validation repository root",
        must_exist=True,
    )
    detached = _exact_path(
        detached_root,
        label="Historical source-validation detached root",
        must_exist=True,
    )
    _require(
        _historical_cwd_authority_mode(
            repository_root=root,
            detached_root=detached,
        )
        == "canonical-root",
        "Historical source validation requires active canonical-root authority.",
    )
    saved_authority = _HISTORICAL_ROOT_CWD_AUTHORITY.get()
    capability = _get_installed_historical_thread_fs_capability()
    _require(
        type(saved_authority) is tuple
        and len(saved_authority) == 5
        and capability is not None
        and saved_authority[0] is capability,
        "Historical source validation lacks the installed root capability.",
    )
    exact_capability = cast(_HistoricalThreadFsIsolationCapability, capability)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Historical source validation requires O_NOFOLLOW.")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0) | cast(int, no_follow)
    root_fd = os.open(".", flags)
    detached_fd = -1
    try:
        detached_fd = os.open(detached, flags)
        root_identity = _directory_identity(os.fstat(root_fd))
        detached_identity = _directory_identity(os.fstat(detached_fd))
        _require(
            root_identity == exact_capability.root_identity
            and detached_identity == exact_capability.detached_identity
            and _cwd_matches(root, root_identity),
            "Historical source-validation directory identities drifted.",
        )
        entered_detached = False
        cleanup_errors: list[BaseException] = []
        try:
            os.fchdir(detached_fd)
            entered_detached = True
            _assert_thread_fs_isolation_capability(
                repository_root=root,
                detached_root=detached,
                main_location="detached",
            )
            yield
        finally:
            detached_held = entered_detached and _cwd_matches(detached, detached_identity)
            root_authority_held = _HISTORICAL_ROOT_CWD_AUTHORITY.get() is saved_authority
            if entered_detached:
                try:
                    _assert_thread_fs_isolation_capability(
                        repository_root=root,
                        detached_root=detached,
                        main_location="detached",
                    )
                except BaseException as error:
                    cleanup_errors.append(error)
            try:
                os.fchdir(root_fd)
            except BaseException as error:
                cleanup_errors.append(error)
            if _HISTORICAL_ROOT_CWD_AUTHORITY.get() is not saved_authority:
                try:
                    _HISTORICAL_ROOT_CWD_AUTHORITY.set(saved_authority)
                except BaseException as error:
                    cleanup_errors.append(error)
            authority_restored = _HISTORICAL_ROOT_CWD_AUTHORITY.get() is saved_authority
            root_restored = _cwd_matches(root, root_identity)
            if root_restored and authority_restored:
                try:
                    _assert_thread_fs_isolation_capability(
                        repository_root=root,
                        detached_root=detached,
                        main_location="canonical-root",
                    )
                except BaseException as error:
                    cleanup_errors.append(error)
            if not (
                entered_detached
                and detached_held
                and root_authority_held
                and root_restored
                and authority_restored
            ):
                cleanup_errors.append(
                    ValueError("Historical source-validation cwd or authority restoration drifted.")
                )
            if cleanup_errors:
                raise ValueError(
                    "Historical source-validation cleanup failed after restoration attempts."
                ) from cleanup_errors[0]
    finally:
        if detached_fd >= 0:
            os.close(detached_fd)
        os.close(root_fd)


@contextmanager
def _legacy_training_command_cwd_adapter(
    training_matrix: Any,
    *,
    repository_root: Path,
    detached_root: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    raw_command_original = getattr(training_matrix, "_validate_training_command", None)
    raw_checkpoint_original = getattr(training_matrix, "_validate_checkpoint", None)
    _require(
        callable(raw_command_original)
        and getattr(raw_command_original, "__name__", None) == "_validate_training_command"
        and callable(raw_checkpoint_original)
        and getattr(raw_checkpoint_original, "__name__", None) == "_validate_checkpoint",
        "Legacy training relative-path validators are unavailable or aliased.",
    )
    command_original = cast(Callable[..., Any], raw_command_original)
    checkpoint_original = cast(Callable[..., Any], raw_checkpoint_original)
    rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in inventory:
        key = (cast(str, row.get("scale")), cast(int, row.get("training_seed")))
        _require(
            key[0] in SCALES and key[1] in TRAINING_SEEDS and key not in rows,
            "Legacy training command adapter inventory is invalid.",
        )
        rows[key] = row
    expected = {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS}
    _require(set(rows) == expected, "Legacy training command adapter inventory is incomplete.")
    validated_commands: dict[tuple[str, int], int] = {}
    validated_checkpoints: dict[tuple[str, int], int] = {}
    validated_command_profiles: dict[str, int] = {}
    validated_checkpoint_profiles: dict[str, int] = {}
    validated = {
        "training_command": validated_commands,
        "training_checkpoint": validated_checkpoints,
        "training_command_profiles": validated_command_profiles,
        "training_checkpoint_profiles": validated_checkpoint_profiles,
    }

    def adapter(command: Any, **arguments: Any) -> Any:
        scale = arguments.get("scale")
        seed = arguments.get("seed")
        key = (cast(str, scale), cast(int, seed))
        _require(
            key in rows,
            "Legacy training command adapter call drifted.",
        )
        row = rows[key]
        output_dir = arguments.get("output_dir")
        _historical_cwd_authority_mode(
            repository_root=repository_root,
            detached_root=detached_root,
        )
        path_spelling_mode = _historical_training_path_spelling_mode()
        expected_output_dir = (
            Path(cast(str, row.get("resolved_path")))
            if path_spelling_mode == "absolute"
            else Path(cast(str, row.get("recorded_relative_path")))
        )
        context = arguments.get("context")
        expected_context = row.get("expected_context_profile")
        _require(
            _historical_command_option(
                command,
                "--output-dir",
                label=f"Legacy training command {key[0]}/{key[1]}",
            )
            == row.get("recorded_relative_path")
            and isinstance(output_dir, Path)
            and output_dir == expected_output_dir,
            "Legacy training command adapter arguments differ from signed preflight.",
        )
        _require(
            isinstance(expected_context, Mapping)
            and getattr(context, "manifest_binding", None)
            == expected_context.get("manifest_binding")
            and getattr(context, "source", None) == expected_context.get("source"),
            "Legacy training command context differs from its signed coordinate profile.",
        )
        profile_sha256 = _json_digest(
            {
                "coordinate": {"scale": key[0], "training_seed": key[1]},
                "command_sha256": row.get("command_sha256"),
                "context_profile_sha256": _json_digest(expected_context),
                "path_spelling_mode": path_spelling_mode,
            }
        )
        with _scoped_historical_command_resolution_cwd(
            repository_root=repository_root,
            detached_root=detached_root,
        ):
            result = command_original(command, **arguments)
        validated_commands[key] = validated_commands.get(key, 0) + 1
        validated_command_profiles[profile_sha256] = (
            validated_command_profiles.get(profile_sha256, 0) + 1
        )
        return result

    def checkpoint_adapter(checkpoint: Any, **arguments: Any) -> Any:
        scale = arguments.get("scale")
        seed = arguments.get("seed")
        key = (cast(str, scale), cast(int, seed))
        _require(
            key in rows,
            "Legacy training checkpoint adapter call drifted.",
        )
        row = rows[key]
        expected_path = arguments.get("expected_path")
        _historical_cwd_authority_mode(
            repository_root=repository_root,
            detached_root=detached_root,
        )
        path_spelling_mode = _historical_training_path_spelling_mode()
        expected_checkpoint_path = (
            Path(cast(str, row.get("resolved_checkpoint_path")))
            if path_spelling_mode == "absolute"
            else Path(cast(str, row.get("checkpoint_relative_path")))
        )
        context = arguments.get("context")
        expected_context = row.get("expected_context_profile")
        return_raw = arguments.get("return_raw", False)
        _require(
            isinstance(checkpoint, dict)
            and checkpoint == row.get("checkpoint_binding")
            and isinstance(expected_path, Path)
            and expected_path == expected_checkpoint_path,
            "Legacy training checkpoint adapter arguments differ from signed preflight.",
        )
        _require(
            isinstance(expected_context, Mapping)
            and getattr(context, "manifest_binding", None)
            == expected_context.get("manifest_binding")
            and getattr(context, "source", None) == expected_context.get("source")
            and type(return_raw) is bool,
            "Legacy training checkpoint context or raw mode differs from signed preflight.",
        )
        profile_sha256 = _json_digest(
            {
                "coordinate": {"scale": key[0], "training_seed": key[1]},
                "context_profile_sha256": _json_digest(expected_context),
                "path_spelling_mode": path_spelling_mode,
                "return_raw": cast(bool, return_raw),
            }
        )
        with _scoped_historical_command_resolution_cwd(
            repository_root=repository_root,
            detached_root=detached_root,
        ):
            result = checkpoint_original(checkpoint, **arguments)
        validated_checkpoints[key] = validated_checkpoints.get(key, 0) + 1
        validated_checkpoint_profiles[profile_sha256] = (
            validated_checkpoint_profiles.get(profile_sha256, 0) + 1
        )
        return result

    training_matrix._validate_training_command = adapter
    training_matrix._validate_checkpoint = checkpoint_adapter
    try:
        yield validated
    finally:
        adapters_held = (
            getattr(training_matrix, "_validate_training_command", None) is adapter
            and getattr(training_matrix, "_validate_checkpoint", None) is checkpoint_adapter
        )
        training_matrix._validate_checkpoint = checkpoint_original
        training_matrix._validate_training_command = command_original
        _require(
            adapters_held
            and getattr(training_matrix, "_validate_training_command", None) is command_original
            and getattr(training_matrix, "_validate_checkpoint", None) is checkpoint_original,
            "Legacy training relative-path validator adapter identity drifted.",
        )


def _historical_superseded_bundle_argument_claim(repository_root: Path) -> dict[str, Any]:
    root = _exact_path(
        repository_root,
        label="Historical superseded-bundle repository root",
        must_exist=True,
    )
    payload, opened = _load_json_nofollow(
        root / HISTORICAL_TRAINING_LEDGER,
        label="Historical superseded-bundle training ledger",
    )
    try:
        trainer_binding = payload.get("canonical_trainer")
        execution_environment = payload.get("execution_environment")
        _require(
            opened.sha256 == HISTORICAL_TRAINING_LEDGER_SHA256
            and isinstance(trainer_binding, Mapping)
            and isinstance(execution_environment, Mapping),
            "Historical superseded-bundle argument evidence is invalid.",
        )
        result = {
            "schema_version": 1,
            "function": "training_matrix._validate_superseded_training_bundle",
            "output_root": str(root / HISTORICAL_TRAINING_ROOT),
            "trainer_binding": dict(cast(Mapping[str, Any], trainer_binding)),
            "signed_execution_environment": dict(cast(Mapping[str, Any], execution_environment)),
            "allowed_return_raw_checkpoint": [False, True],
        }
        opened.assert_unchanged()
        return result
    finally:
        opened.close()


def _historical_cwd_authority_mode(
    *,
    repository_root: Path,
    detached_root: Path,
) -> str:
    authority = _HISTORICAL_ROOT_CWD_AUTHORITY.get()
    if authority is None:
        _assert_thread_fs_isolation_capability(
            repository_root=repository_root,
            detached_root=detached_root,
            main_location="detached",
        )
        return "detached"
    _require(
        type(authority) is tuple
        and len(authority) == 5
        and authority[1] == repository_root
        and authority[2] == detached_root,
        "Historical nested cwd authority is malformed or bound to different roots.",
    )
    capability = _assert_thread_fs_isolation_capability(
        repository_root=repository_root,
        detached_root=detached_root,
        main_location="canonical-root",
    )
    _require(
        authority[0] is capability
        and authority[3] == capability.root_identity
        and authority[4] == capability.detached_identity,
        "Historical nested cwd authority does not hold the installed capability.",
    )
    return "canonical-root"


def _historical_training_path_spelling_mode() -> str:
    authority = _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get()
    if authority is None:
        return "absolute"
    installed = _get_installed_historical_thread_fs_capability()
    _require(
        type(authority) is _HistoricalSupersededPathSpellingAuthority
        and authority.seal is _HISTORICAL_SUPERSEDED_PATH_SPELLING_SEAL
        and installed is not None
        and authority.capability is installed
        and _historical_thread_fs_capability_is_installed(authority.capability)
        and authority.mode in {"absolute", "recorded-relative"},
        "Historical training path-spelling authority is forged or invalid.",
    )
    return authority.mode


@contextmanager
def _legacy_superseded_bundle_cwd_adapter(
    training_matrix: Any,
    *,
    repository_root: Path,
    detached_root: Path,
    trust_root: attestation.TrustRoot,
    argument_claim: Mapping[str, Any],
) -> Iterator[dict[str, int]]:
    raw_original = getattr(training_matrix, "_validate_superseded_training_bundle", None)
    _require(
        callable(raw_original)
        and getattr(raw_original, "__name__", None) == "_validate_superseded_training_bundle"
        and not bool(getattr(raw_original, "_adaptive_v4_superseded_bundle_adapter", False)),
        "Legacy superseded-bundle validator is unavailable, aliased, or already adapted.",
    )
    original = cast(Callable[..., Any], raw_original)
    expected_keys = {
        "output_root",
        "trust_root",
        "trainer_binding",
        "expected_execution_environment",
        "return_raw_checkpoint",
    }
    invocations: dict[str, int] = {}

    def adapter(*arguments: Any, **keywords: Any) -> Any:
        entry_mode = _historical_cwd_authority_mode(
            repository_root=repository_root,
            detached_root=detached_root,
        )
        output_root = keywords.get("output_root")
        absolute_output_root = Path(cast(str, argument_claim.get("output_root")))
        path_spelling_mode = (
            "absolute"
            if isinstance(output_root, Path) and output_root == absolute_output_root
            else "recorded-relative"
            if entry_mode == "canonical-root"
            and isinstance(output_root, Path)
            and output_root == HISTORICAL_TRAINING_ROOT
            else "invalid"
        )
        _require(
            not arguments
            and set(keywords) == expected_keys
            and type(keywords.get("return_raw_checkpoint")) is bool
            and keywords.get("return_raw_checkpoint")
            in argument_claim.get("allowed_return_raw_checkpoint", ())
            and path_spelling_mode in {"absolute", "recorded-relative"}
            and keywords.get("trust_root") is trust_root
            and keywords.get("trainer_binding") == argument_claim.get("trainer_binding")
            and keywords.get("expected_execution_environment")
            in (None, argument_claim.get("signed_execution_environment"))
            and threading.current_thread() is threading.main_thread()
            and threading.active_count() == 1
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy superseded-bundle adapter arguments, thread, or detached source drifted.",
        )
        invocation = {
            "return_raw_checkpoint": cast(bool, keywords["return_raw_checkpoint"]),
            "execution_environment_mode": (
                "none"
                if keywords["expected_execution_environment"] is None
                else "exact-signed-training-ledger"
            ),
            "path_spelling_mode": path_spelling_mode,
        }
        invocation_sha256 = _json_digest(invocation)
        capability = _get_installed_historical_thread_fs_capability()
        _require(
            capability is not None
            and _historical_thread_fs_capability_is_installed(capability)
            and _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is None,
            "Legacy superseded-bundle path-spelling authority is already active.",
        )
        spelling_authority = _HistoricalSupersededPathSpellingAuthority(
            seal=_HISTORICAL_SUPERSEDED_PATH_SPELLING_SEAL,
            capability=cast(_HistoricalThreadFsIsolationCapability, capability),
            mode=path_spelling_mode,
        )
        entry_token = _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.set(spelling_authority)
        cleanup_errors: list[BaseException] = []
        authority_held = False
        try:
            with _scoped_historical_command_resolution_cwd(
                repository_root=repository_root,
                detached_root=detached_root,
            ):
                result = original(**keywords)
        finally:
            authority_held = (
                _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is spelling_authority
            )
            try:
                _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.reset(entry_token)
            except BaseException as error:
                cleanup_errors.append(error)
            if _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is not None:
                try:
                    _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.set(None)
                except BaseException as error:
                    cleanup_errors.append(error)
            authority_cleared = _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is None
            if not (authority_held and authority_cleared):
                cleanup_errors.append(
                    ValueError("Legacy superseded-bundle path-spelling authority drifted.")
                )
            if cleanup_errors:
                raise ValueError(
                    "Legacy superseded-bundle path-spelling authority cleanup failed."
                ) from cleanup_errors[0]
        _require(
            _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == entry_mode
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy superseded-bundle adapter did not restore the detached source.",
        )
        invocations[invocation_sha256] = invocations.get(invocation_sha256, 0) + 1
        return result

    adapter._adaptive_v4_superseded_bundle_adapter = True  # type: ignore[attr-defined]
    training_matrix._validate_superseded_training_bundle = adapter
    try:
        yield invocations
    finally:
        adapter_held = (
            getattr(training_matrix, "_validate_superseded_training_bundle", None) is adapter
        )
        training_matrix._validate_superseded_training_bundle = original
        _require(
            adapter_held
            and getattr(training_matrix, "_validate_superseded_training_bundle", None) is original
            and _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is None
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "detached"
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy superseded-bundle callable identity or detached source drifted.",
        )


def _historical_ledger_record_inventory(
    records: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS}
    _require(
        set(records) == expected
        and all(isinstance(record, Mapping) for record in records.values()),
        "Historical training ledger-record inventory is incomplete.",
    )
    return [
        {
            "scale": scale,
            "training_seed": seed,
            "record": dict(records[(scale, seed)]),
        }
        for scale in SCALES
        for seed in TRAINING_SEEDS
    ]


def _historical_quarantine_argument_claim(repository_root: Path) -> dict[str, Any]:
    root = _exact_path(
        repository_root,
        label="Historical quarantine repository root",
        must_exist=True,
    )
    training_payload, training_opened = _load_json_nofollow(
        root / HISTORICAL_TRAINING_LEDGER,
        label="Historical quarantine training ledger",
    )
    try:
        runs = training_payload.get("runs")
        manifest = training_payload.get("manifest")
        trainer = training_payload.get("canonical_trainer")
        _require(
            training_opened.sha256 == HISTORICAL_TRAINING_LEDGER_SHA256
            and isinstance(runs, list)
            and len(runs) == len(SCALES) * len(TRAINING_SEEDS)
            and all(isinstance(run, Mapping) for run in runs)
            and isinstance(manifest, Mapping)
            and manifest.get("path") == str(root / V1_1_MANIFEST_RELATIVE_PATH)
            and manifest.get("sha256") == V1_1_MANIFEST_SHA256
            and isinstance(trainer, Mapping),
            "Historical quarantine signed argument evidence is invalid.",
        )
        records: dict[tuple[str, int], Mapping[str, Any]] = {}
        for raw_run in cast(list[Any], runs):
            run = cast(Mapping[str, Any], raw_run)
            coordinate = (run.get("scale"), run.get("seed"))
            _require(
                coordinate[0] in SCALES
                and coordinate[1] in TRAINING_SEEDS
                and coordinate not in records,
                "Historical quarantine training run coordinate is invalid.",
            )
            records[cast(tuple[str, int], coordinate)] = run
        record_inventory = _historical_ledger_record_inventory(records)
        invocation_profile = {
            "function": "calibration_matrix._validate_quarantine_evidence",
            "keyword_names": [
                "ledger_records",
                "legacy_context",
                "trainer_binding",
                "training_matrix_payload",
                "training_matrix_summary_path",
                "trust_root",
            ],
            "cwd_translation": "detached-to-canonical-root-to-detached",
            "training_ledger_mode": "exact-signed-v1.1-ledger",
            "legacy_context_mode": "exact-signed-v1.1-manifest-and-source",
        }
        result = {
            "schema_version": 1,
            "function": "calibration_matrix._validate_quarantine_evidence",
            "quarantine_root": str(root / HISTORICAL_CALIBRATION_QUARANTINE_ROOT),
            "training_matrix_summary_path": str(root / HISTORICAL_TRAINING_LEDGER),
            "training_matrix_payload_sha256": _json_digest(training_payload),
            "trainer_binding": dict(cast(Mapping[str, Any], trainer)),
            "ledger_record_count": len(record_inventory),
            "ledger_record_inventory_sha256": _json_digest(
                {"schema_version": 1, "records": record_inventory}
            ),
            "legacy_source": {"commit": V1_1_RESULT_SOURCE_COMMIT, "dirty": False},
            "legacy_manifest": dict(cast(Mapping[str, Any], manifest)),
            "expected_invocation_count": 1,
            "expected_invocation_profile_sha256": _json_digest(invocation_profile),
        }
        training_opened.assert_unchanged()
        return result
    finally:
        training_opened.close()


def _verify_historical_quarantine_argument_claim(raw: object) -> None:
    _require(
        isinstance(raw, Mapping)
        and set(raw)
        == {
            "schema_version",
            "function",
            "quarantine_root",
            "training_matrix_summary_path",
            "training_matrix_payload_sha256",
            "trainer_binding",
            "ledger_record_count",
            "ledger_record_inventory_sha256",
            "legacy_source",
            "legacy_manifest",
            "expected_invocation_count",
            "expected_invocation_profile_sha256",
        }
        and raw.get("schema_version") == 1
        and raw.get("function") == "calibration_matrix._validate_quarantine_evidence"
        and isinstance(raw.get("quarantine_root"), str)
        and Path(cast(str, raw.get("quarantine_root"))).is_absolute()
        and isinstance(raw.get("training_matrix_summary_path"), str)
        and Path(cast(str, raw.get("training_matrix_summary_path"))).is_absolute()
        and _is_sha256(raw.get("training_matrix_payload_sha256"))
        and isinstance(raw.get("trainer_binding"), Mapping)
        and raw.get("ledger_record_count") == len(SCALES) * len(TRAINING_SEEDS)
        and _is_sha256(raw.get("ledger_record_inventory_sha256"))
        and raw.get("legacy_source") == {"commit": V1_1_RESULT_SOURCE_COMMIT, "dirty": False}
        and isinstance(raw.get("legacy_manifest"), Mapping)
        and cast(Mapping[str, Any], raw.get("legacy_manifest")).get("sha256")
        == V1_1_MANIFEST_SHA256
        and raw.get("expected_invocation_count") == 1
        and _is_sha256(raw.get("expected_invocation_profile_sha256")),
        "Historical quarantine cwd-adapter claim drifted.",
    )


def _historical_relative_path_adapter_claim(
    training_inventory: Sequence[Mapping[str, Any]],
    superseded_argument_claim: Mapping[str, Any],
    quarantine_argument_claim: Mapping[str, Any],
    *,
    observed_training: Mapping[str, Any] | None = None,
    observed_superseded: Mapping[str, int] | None = None,
    observed_quarantine: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in training_inventory]
    _require(
        len(rows) == len(SCALES) * len(TRAINING_SEEDS),
        "Historical relative-path adapter training inventory is incomplete.",
    )
    command_profiles: dict[str, int] = {}
    checkpoint_profiles: dict[str, int] = {}
    coordinate_counts: dict[tuple[str, int], int] = {}
    for row in rows:
        coordinate = (row.get("scale"), row.get("training_seed"))
        context_profile = row.get("expected_context_profile")
        _require(
            coordinate[0] in SCALES
            and coordinate[1] in TRAINING_SEEDS
            and isinstance(context_profile, Mapping)
            and isinstance(context_profile.get("manifest_binding"), Mapping)
            and isinstance(context_profile.get("source"), Mapping)
            and _is_sha256(row.get("command_sha256")),
            "Historical relative-path adapter coordinate context is invalid.",
        )
        key = cast(tuple[str, int], coordinate)
        _require(key not in coordinate_counts, "Historical relative-path coordinate is duplicated.")
        invocation_count = 71 if key == (SCALES[0], TRAINING_SEEDS[0]) else 69
        coordinate_counts[key] = invocation_count
        context_sha256 = _json_digest(context_profile)
        command_mode_counts = (
            {"absolute": 70, "recorded-relative": 1}
            if key == (SCALES[0], TRAINING_SEEDS[0])
            else {"absolute": 69}
        )
        for path_spelling_mode, count in command_mode_counts.items():
            command_sha256 = _json_digest(
                {
                    "coordinate": {"scale": key[0], "training_seed": key[1]},
                    "command_sha256": row["command_sha256"],
                    "context_profile_sha256": context_sha256,
                    "path_spelling_mode": path_spelling_mode,
                }
            )
            command_profiles[command_sha256] = count
        checkpoint_modes = (
            ((False, "absolute", 62), (True, "absolute", 8), (True, "recorded-relative", 1))
            if key == (SCALES[0], TRAINING_SEEDS[0])
            else ((False, "absolute", 62), (True, "absolute", 7))
        )
        for return_raw, path_spelling_mode, count in checkpoint_modes:
            checkpoint_sha256 = _json_digest(
                {
                    "coordinate": {"scale": key[0], "training_seed": key[1]},
                    "context_profile_sha256": context_sha256,
                    "path_spelling_mode": path_spelling_mode,
                    "return_raw": return_raw,
                }
            )
            checkpoint_profiles[checkpoint_sha256] = count
    superseded_profiles = {
        _json_digest(
            {
                "return_raw_checkpoint": False,
                "execution_environment_mode": "exact-signed-training-ledger",
                "path_spelling_mode": "absolute",
            }
        ): 62,
        _json_digest(
            {
                "return_raw_checkpoint": True,
                "execution_environment_mode": "none",
                "path_spelling_mode": "absolute",
            }
        ): 1,
        _json_digest(
            {
                "return_raw_checkpoint": True,
                "execution_environment_mode": "none",
                "path_spelling_mode": "recorded-relative",
            }
        ): 1,
        _json_digest(
            {
                "return_raw_checkpoint": True,
                "execution_environment_mode": "exact-signed-training-ledger",
                "path_spelling_mode": "absolute",
            }
        ): 7,
    }
    _require(
        superseded_argument_claim.get("function")
        == "training_matrix._validate_superseded_training_bundle"
        and quarantine_argument_claim.get("function")
        == "calibration_matrix._validate_quarantine_evidence"
        and quarantine_argument_claim.get("expected_invocation_count") == 1
        and _is_sha256(quarantine_argument_claim.get("expected_invocation_profile_sha256")),
        "Historical relative-path superseded or quarantine claim is invalid.",
    )
    quarantine_profiles = {
        cast(str, quarantine_argument_claim["expected_invocation_profile_sha256"]): 1
    }
    if observed_training is not None:
        _require(
            observed_training.get("training_command") == coordinate_counts
            and observed_training.get("training_checkpoint") == coordinate_counts
            and observed_training.get("training_command_profiles") == command_profiles
            and observed_training.get("training_checkpoint_profiles") == checkpoint_profiles,
            "Historical training relative-path invocation multiset drifted.",
        )
    if observed_superseded is not None:
        _require(
            dict(observed_superseded) == superseded_profiles,
            "Historical superseded-bundle invocation multiset drifted.",
        )
    if observed_quarantine is not None:
        _require(
            dict(observed_quarantine) == quarantine_profiles,
            "Historical quarantine invocation multiset drifted.",
        )

    def function_claim(
        function: str,
        profiles: Mapping[str, int],
    ) -> dict[str, Any]:
        invocation_rows = [
            {"profile_sha256": profile_sha256, "invocations": count}
            for profile_sha256, count in sorted(profiles.items())
        ]
        return {
            "function": function,
            "expected_profile_count": len(invocation_rows),
            "expected_invocation_count": sum(profiles.values()),
            "expected_invocation_multiset_sha256": _json_digest(
                {"schema_version": 1, "invocations": invocation_rows}
            ),
        }

    functions = [
        function_claim("calibration_matrix._validate_quarantine_evidence", quarantine_profiles),
        function_claim("training_matrix._validate_checkpoint", checkpoint_profiles),
        function_claim("training_matrix._validate_superseded_training_bundle", superseded_profiles),
        function_claim("training_matrix._validate_training_command", command_profiles),
    ]
    return {
        "schema_version": 1,
        "cwd_isolation": "CLONE_FS-private-main-task-exact-directory-fd-scopes",
        "callable_identity_restored": True,
        "exact_invocation_multiplicities_required": True,
        "functions": functions,
        "expected_invocation_count": sum(
            cast(int, function["expected_invocation_count"]) for function in functions
        ),
    }


def _verify_historical_relative_path_adapter_claim(raw: object) -> None:
    _require(
        isinstance(raw, Mapping)
        and set(raw)
        == {
            "schema_version",
            "cwd_isolation",
            "callable_identity_restored",
            "exact_invocation_multiplicities_required",
            "functions",
            "expected_invocation_count",
        }
        and raw.get("schema_version") == 1
        and raw.get("cwd_isolation") == "CLONE_FS-private-main-task-exact-directory-fd-scopes"
        and raw.get("callable_identity_restored") is True
        and raw.get("exact_invocation_multiplicities_required") is True
        and raw.get("expected_invocation_count") == 1456
        and isinstance(raw.get("functions"), list),
        "Historical relative-path adapter invocation claim drifted.",
    )
    functions = cast(Mapping[str, Any], raw).get("functions")
    rows = [cast(Mapping[str, Any], row) for row in cast(list[Any], functions)]
    _require(
        len(rows) == 4
        and all(
            isinstance(row, Mapping)
            and set(row)
            == {
                "function",
                "expected_profile_count",
                "expected_invocation_count",
                "expected_invocation_multiset_sha256",
            }
            and _is_sha256(row.get("expected_invocation_multiset_sha256"))
            for row in rows
        )
        and [
            (
                row.get("function"),
                row.get("expected_profile_count"),
                row.get("expected_invocation_count"),
            )
            for row in rows
        ]
        == [
            ("calibration_matrix._validate_quarantine_evidence", 1, 1),
            ("training_matrix._validate_checkpoint", 21, 692),
            ("training_matrix._validate_superseded_training_bundle", 4, 71),
            ("training_matrix._validate_training_command", 11, 692),
        ],
        "Historical relative-path adapter function multiset claim drifted.",
    )


def _historical_retry_admission_argument_profile(
    *,
    path: Path,
    output_root: Path,
    context: Any,
    evidence: Any,
    incident_report_binding: Mapping[str, Any],
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    manifest_binding = getattr(context, "manifest_binding", None)
    source = getattr(context, "source", None)
    manifest_path = getattr(context, "manifest_path", None)
    legacy_context = getattr(evidence, "legacy_context", None)
    legacy_manifest_binding = getattr(legacy_context, "manifest_binding", None)
    legacy_source = getattr(legacy_context, "source", None)
    legacy_manifest_path = getattr(legacy_context, "manifest_path", None)
    matrix_ledger = getattr(evidence, "matrix_ledger", None)
    _require(
        isinstance(path, Path)
        and path.is_absolute()
        and isinstance(output_root, Path)
        and output_root.is_absolute()
        and isinstance(manifest_path, Path)
        and manifest_path.is_absolute()
        and isinstance(manifest_binding, Mapping)
        and isinstance(source, Mapping)
        and isinstance(legacy_manifest_path, Path)
        and legacy_manifest_path.is_absolute()
        and isinstance(legacy_manifest_binding, Mapping)
        and isinstance(legacy_source, Mapping)
        and isinstance(matrix_ledger, Mapping)
        and isinstance(matrix_ledger.get("gpu_lease"), Mapping)
        and isinstance(incident_report_binding, Mapping)
        and type(trust_root) is attestation.TrustRoot,
        "Historical retry-admission semantic arguments are malformed.",
    )

    def evidence_mapping(name: str) -> dict[str, Any]:
        value = getattr(evidence, name, None)
        _require(
            isinstance(value, Mapping),
            f"Historical retry-admission evidence {name} is malformed.",
        )
        return dict(cast(Mapping[str, Any], value))

    matrix_ledger_map = cast(Mapping[str, Any], matrix_ledger)
    return {
        "path": str(path),
        "output_root": str(output_root),
        "context": {
            "manifest_path": str(manifest_path),
            "manifest_binding": dict(cast(Mapping[str, Any], manifest_binding)),
            "source": dict(cast(Mapping[str, Any], source)),
        },
        "evidence": {
            "legacy_context": {
                "manifest_path": str(legacy_manifest_path),
                "manifest_binding": dict(cast(Mapping[str, Any], legacy_manifest_binding)),
                "source": dict(cast(Mapping[str, Any], legacy_source)),
            },
            "legacy_manifest_file_binding": evidence_mapping("legacy_manifest_file_binding"),
            "matrix_ledger_binding": evidence_mapping("matrix_ledger_binding"),
            "claim_binding": evidence_mapping("claim_binding"),
            "artifact_binding": evidence_mapping("artifact_binding"),
            "training_matrix_binding": evidence_mapping("training_matrix_binding"),
            "checkpoint_binding": evidence_mapping("checkpoint_binding"),
            "execution_environment": evidence_mapping("execution_environment"),
            "matrix_ledger_gpu_lease": dict(
                cast(Mapping[str, Any], matrix_ledger_map["gpu_lease"])
            ),
        },
        "incident_report_binding": dict(incident_report_binding),
        "trust_root_key_id": trust_root.key_id,
    }


def _historical_retry_admission_cwd_claim(repository_root: Path) -> dict[str, Any]:
    root = _exact_path(
        repository_root,
        label="Historical retry-admission repository root",
        must_exist=True,
    )
    admission_path = root / HISTORICAL_CALIBRATION_ADMISSION
    payload, opened = _load_json_nofollow(
        admission_path,
        label="Historical retry admission",
        require_canonical_pretty_bytes=True,
    )
    try:
        current_manifest = payload.get("current_manifest")
        current_source = payload.get("current_source")
        superseded_manifest = payload.get("superseded_manifest")
        retry_rule = payload.get("retry_rule")
        incident_report = payload.get("incident_report")
        envelope = payload.get("attestation")
        attested_semantic = dict(payload)
        attested_semantic.pop("attestation", None)
        _require(
            opened.sha256 == HISTORICAL_CALIBRATION_ADMISSION_SHA256
            and opened.bytes == HISTORICAL_CALIBRATION_ADMISSION_BYTES
            and payload.get("schema_version") == 1
            and payload.get("admission_id") == "p2-direct-calibration-one-shot-retry-admission-v1.2"
            and payload.get("status") == "terminal"
            and isinstance(current_manifest, Mapping)
            and isinstance(current_source, Mapping)
            and isinstance(superseded_manifest, Mapping)
            and isinstance(retry_rule, Mapping)
            and isinstance(retry_rule.get("failed_attempt_gpu_lease"), Mapping)
            and isinstance(incident_report, Mapping)
            and isinstance(envelope, Mapping)
            and envelope.get("purpose") == LEGACY_RETRY_ADMISSION_PURPOSE
            and _is_sha256(envelope.get("key_id"))
            and _is_sha256(envelope.get("mac"))
            and envelope.get("payload_sha256") == _json_digest(attested_semantic),
            "Historical retry-admission artifact binding is invalid.",
        )
        base_semantic = dict(payload)
        base_semantic.pop("attestation")
        base_semantic.pop("payload_sha256")
        _require(
            payload.get("payload_sha256") == _json_digest(base_semantic),
            "Historical retry-admission payload digest drifted.",
        )

        def without_bytes(value: Mapping[str, Any]) -> dict[str, Any]:
            result = dict(value)
            result.pop("bytes", None)
            return result

        current_manifest_map = cast(Mapping[str, Any], current_manifest)
        superseded_manifest_map = cast(Mapping[str, Any], superseded_manifest)
        retry_rule_map = cast(Mapping[str, Any], retry_rule)
        expected_profile = {
            "path": str(admission_path),
            "output_root": str(root / HISTORICAL_CALIBRATION_ROOT),
            "context": {
                "manifest_path": current_manifest_map["path"],
                "manifest_binding": without_bytes(current_manifest_map),
                "source": dict(cast(Mapping[str, Any], current_source)),
            },
            "evidence": {
                "legacy_context": {
                    "manifest_path": superseded_manifest_map["path"],
                    "manifest_binding": without_bytes(superseded_manifest_map),
                    "source": {"commit": V1_1_RESULT_SOURCE_COMMIT, "dirty": False},
                },
                "legacy_manifest_file_binding": {
                    "path": superseded_manifest_map["path"],
                    "sha256": superseded_manifest_map["sha256"],
                    "bytes": superseded_manifest_map["bytes"],
                },
                "matrix_ledger_binding": dict(
                    cast(Mapping[str, Any], payload["superseded_matrix_ledger"])
                ),
                "claim_binding": dict(cast(Mapping[str, Any], payload["preserved_claim"])),
                "artifact_binding": dict(
                    cast(Mapping[str, Any], payload["preserved_calibration_artifact"])
                ),
                "training_matrix_binding": dict(
                    cast(Mapping[str, Any], payload["terminal_training_matrix_ledger"])
                ),
                "checkpoint_binding": dict(cast(Mapping[str, Any], payload["checkpoint"])),
                "execution_environment": dict(
                    cast(Mapping[str, Any], payload["execution_environment"])
                ),
                "matrix_ledger_gpu_lease": dict(
                    cast(Mapping[str, Any], retry_rule_map["failed_attempt_gpu_lease"])
                ),
            },
            "incident_report_binding": dict(cast(Mapping[str, Any], incident_report)),
            "trust_root_key_id": cast(Mapping[str, Any], envelope)["key_id"],
        }
        public_binding = {
            "path": str(admission_path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
            "payload_sha256": payload["payload_sha256"],
            "attestation_mac": cast(Mapping[str, Any], envelope)["mac"],
            "admission_id": payload["admission_id"],
            "coordinate": payload["coordinate"],
            "preserved_claim_sha256": cast(Mapping[str, Any], payload["preserved_claim"])["sha256"],
            "preserved_artifact_sha256": cast(
                Mapping[str, Any], payload["preserved_calibration_artifact"]
            )["sha256"],
        }
        result = {
            "schema_version": 1,
            "function": "calibration_matrix._load_retry_admission",
            "cwd_translation": "detached-to-canonical-root-to-detached",
            "cwd_sensitive_semantic_field": "quarantine_rule.root",
            "expected_canonical_quarantine_root": str(
                root / HISTORICAL_CALIBRATION_QUARANTINE_ROOT
            ),
            "retry_admission": {
                "path": str(admission_path),
                "sha256": opened.sha256,
                "bytes": opened.bytes,
                "payload_sha256": payload["payload_sha256"],
            },
            "argument_profile": expected_profile,
            "argument_profile_sha256": _json_digest(expected_profile),
            "expected_result_payload_json_sha256": _json_digest(payload),
            "expected_result_public_binding": public_binding,
            "expected_invocation_count": 1,
            "callable_identity_restored": True,
        }
        opened.assert_unchanged()
        return result
    finally:
        opened.close()


def _verify_historical_retry_admission_cwd_claim(raw: object) -> None:
    _require(
        isinstance(raw, Mapping)
        and set(raw)
        == {
            "schema_version",
            "function",
            "cwd_translation",
            "cwd_sensitive_semantic_field",
            "expected_canonical_quarantine_root",
            "retry_admission",
            "argument_profile",
            "argument_profile_sha256",
            "expected_result_payload_json_sha256",
            "expected_result_public_binding",
            "expected_invocation_count",
            "callable_identity_restored",
        }
        and raw.get("schema_version") == 1
        and raw.get("function") == "calibration_matrix._load_retry_admission"
        and raw.get("cwd_translation") == "detached-to-canonical-root-to-detached"
        and raw.get("cwd_sensitive_semantic_field") == "quarantine_rule.root"
        and raw.get("expected_invocation_count") == 1
        and raw.get("callable_identity_restored") is True
        and isinstance(raw.get("expected_canonical_quarantine_root"), str)
        and Path(cast(str, raw.get("expected_canonical_quarantine_root"))).is_absolute()
        and isinstance(raw.get("retry_admission"), Mapping)
        and isinstance(raw.get("argument_profile"), Mapping)
        and _is_sha256(raw.get("argument_profile_sha256"))
        and raw.get("argument_profile_sha256") == _json_digest(raw.get("argument_profile"))
        and _is_sha256(raw.get("expected_result_payload_json_sha256"))
        and isinstance(raw.get("expected_result_public_binding"), Mapping),
        "Historical retry-admission cwd-adapter claim drifted.",
    )
    claim = cast(Mapping[str, Any], raw)
    retry_binding = cast(Mapping[str, Any], claim["retry_admission"])
    profile = cast(Mapping[str, Any], claim["argument_profile"])
    result_binding = cast(Mapping[str, Any], claim["expected_result_public_binding"])
    _require(
        set(retry_binding) == {"path", "sha256", "bytes", "payload_sha256"}
        and isinstance(retry_binding.get("path"), str)
        and Path(cast(str, retry_binding.get("path"))).is_absolute()
        and retry_binding.get("sha256") == HISTORICAL_CALIBRATION_ADMISSION_SHA256
        and retry_binding.get("bytes") == HISTORICAL_CALIBRATION_ADMISSION_BYTES
        and _is_sha256(retry_binding.get("payload_sha256"))
        and set(profile)
        == {
            "path",
            "output_root",
            "context",
            "evidence",
            "incident_report_binding",
            "trust_root_key_id",
        }
        and profile.get("path") == retry_binding.get("path")
        and isinstance(profile.get("output_root"), str)
        and Path(cast(str, profile.get("output_root"))).is_absolute()
        and isinstance(profile.get("context"), Mapping)
        and isinstance(profile.get("evidence"), Mapping)
        and isinstance(profile.get("incident_report_binding"), Mapping)
        and _is_sha256(profile.get("trust_root_key_id"))
        and set(result_binding)
        == {
            "path",
            "sha256",
            "bytes",
            "payload_sha256",
            "attestation_mac",
            "admission_id",
            "coordinate",
            "preserved_claim_sha256",
            "preserved_artifact_sha256",
        }
        and result_binding.get("path") == retry_binding.get("path")
        and result_binding.get("sha256") == retry_binding.get("sha256")
        and result_binding.get("bytes") == retry_binding.get("bytes")
        and result_binding.get("payload_sha256") == retry_binding.get("payload_sha256"),
        "Historical retry-admission profile or result binding drifted.",
    )


@contextmanager
def _legacy_retry_admission_cwd_adapter(
    calibration_matrix: Any,
    *,
    repository_root: Path,
    detached_root: Path,
    trust_root: attestation.TrustRoot,
    claim: Mapping[str, Any],
) -> Iterator[dict[str, int]]:
    _verify_historical_retry_admission_cwd_claim(claim)
    raw_original = getattr(calibration_matrix, "_load_retry_admission", None)
    training_matrix = getattr(calibration_matrix, "training_matrix", None)
    context_type = getattr(training_matrix, "FrozenContext", None)
    evidence_type = getattr(calibration_matrix, "ValidatedQuarantineEvidence", None)
    result_type = getattr(calibration_matrix, "ValidatedRetryAdmission", None)
    _require(
        callable(raw_original)
        and getattr(raw_original, "__name__", None) == "_load_retry_admission"
        and not bool(getattr(raw_original, "_adaptive_v4_retry_admission_cwd_adapter", False))
        and isinstance(context_type, type)
        and isinstance(evidence_type, type)
        and isinstance(result_type, type),
        "Legacy retry-admission loader or exact result types are unavailable.",
    )
    original = cast(Callable[..., Any], raw_original)
    expected_profile = cast(Mapping[str, Any], claim["argument_profile"])
    expected_profile_sha256 = cast(str, claim["argument_profile_sha256"])
    expected_result_binding = cast(Mapping[str, Any], claim["expected_result_public_binding"])
    attempts = 0
    successes = 0
    observed: dict[str, int] = {}

    def adapter(*arguments: Any, **keywords: Any) -> Any:
        nonlocal attempts, successes
        attempts += 1
        expected_keys = {
            "path",
            "output_root",
            "context",
            "evidence",
            "incident_report_binding",
            "trust_root",
        }
        _require(
            attempts == 1
            and not arguments
            and set(keywords) == expected_keys
            and isinstance(keywords.get("path"), Path)
            and isinstance(keywords.get("output_root"), Path)
            and type(keywords.get("context")) is context_type
            and type(keywords.get("evidence")) is evidence_type
            and isinstance(keywords.get("incident_report_binding"), Mapping)
            and keywords.get("trust_root") is trust_root
            and type(trust_root) is attestation.TrustRoot
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "detached"
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy retry-admission loader arguments or detached authority drifted.",
        )
        path = cast(Path, keywords["path"])
        output_root = cast(Path, keywords["output_root"])
        context = keywords["context"]
        evidence = keywords["evidence"]
        profile = _historical_retry_admission_argument_profile(
            path=path,
            output_root=output_root,
            context=context,
            evidence=evidence,
            incident_report_binding=cast(Mapping[str, Any], keywords["incident_report_binding"]),
            trust_root=trust_root,
        )
        _require(
            profile == expected_profile
            and _json_digest(profile) == expected_profile_sha256
            and path == repository_root / HISTORICAL_CALIBRATION_ADMISSION
            and output_root == repository_root / HISTORICAL_CALIBRATION_ROOT,
            "Legacy retry-admission loader semantic input profile drifted.",
        )
        with _scoped_historical_command_resolution_cwd(
            repository_root=repository_root,
            detached_root=detached_root,
        ):
            result = original(**keywords)
        _require(
            type(result) is result_type
            and getattr(result, "evidence", None) is evidence
            and isinstance(getattr(result, "payload", None), Mapping)
            and _json_digest(result.payload) == claim.get("expected_result_payload_json_sha256")
            and getattr(result, "public_binding", None) == expected_result_binding
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "detached"
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy retry-admission result binding or detached restoration drifted.",
        )
        successes += 1
        observed[expected_profile_sha256] = observed.get(expected_profile_sha256, 0) + 1
        return result

    adapter._adaptive_v4_retry_admission_cwd_adapter = True  # type: ignore[attr-defined]
    calibration_matrix._load_retry_admission = adapter
    try:
        yield observed
    finally:
        adapter_held = getattr(calibration_matrix, "_load_retry_admission", None) is adapter
        calibration_matrix._load_retry_admission = original
        _require(
            adapter_held
            and getattr(calibration_matrix, "_load_retry_admission", None) is original
            and attempts == successes == claim.get("expected_invocation_count") == 1
            and observed == {expected_profile_sha256: 1}
            and _HISTORICAL_ROOT_CWD_AUTHORITY.get() is None
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "detached"
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy retry-admission callable, count, or cwd restoration drifted.",
        )


def _historical_frozen_context_profile(
    repository_root: Path,
    *,
    name: str,
    manifest_relative_path: Path,
    expected_manifest_sha256: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    root = _exact_path(
        repository_root,
        label=f"Historical {name} context repository root",
        must_exist=True,
    )
    manifest_path = root / manifest_relative_path
    payload, opened = _load_json_nofollow(
        manifest_path,
        label=f"Historical {name} context manifest",
    )
    try:
        implementation = payload.get("implementation")
        attestation_binding = payload.get("attestation")
        _require(
            opened.sha256 == expected_manifest_sha256
            and isinstance(implementation, Mapping)
            and _is_sha256(implementation.get("tree_digest"))
            and _is_git_oid(implementation.get("source_commit"))
            and isinstance(attestation_binding, Mapping)
            and attestation_binding.get("key_id")
            == "67f433c02a291f9b1c9e65218171b6da46ef019567ee24406b4738c2ddf765bf"
            and isinstance(payload.get("experiment_id"), str)
            and source.get("dirty") is False
            and _is_git_oid(source.get("commit")),
            f"Historical {name} frozen-context evidence is invalid.",
        )
        implementation_map = cast(Mapping[str, Any], implementation)
        profile = {
            "name": name,
            "manifest_path": str(manifest_path),
            "manifest_binding": {
                "path": str(manifest_path),
                "sha256": opened.sha256,
                "experiment_id": payload["experiment_id"],
                "implementation_digest": implementation_map["tree_digest"],
                "implementation_source_commit": implementation_map["source_commit"],
                "attestation": dict(cast(Mapping[str, Any], attestation_binding)),
            },
            "source": dict(source),
        }
        opened.assert_unchanged()
        return {**profile, "profile_sha256": _json_digest(profile)}
    finally:
        opened.close()


def _historical_calibration_provenance_claim(
    repository_root: Path,
    training_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    root = _exact_path(
        repository_root,
        label="Historical calibration-provenance repository root",
        must_exist=True,
    )
    rows = [dict(row) for row in training_inventory]
    _require(
        len(rows) == len(SCALES) * len(TRAINING_SEEDS),
        "Historical calibration-provenance training inventory is incomplete.",
    )
    current_context = _historical_frozen_context_profile(
        root,
        name="current-v1.2",
        manifest_relative_path=HISTORICAL_MANIFEST_RELATIVE_PATH,
        expected_manifest_sha256=HISTORICAL_MANIFEST_SHA256,
        source={"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
    )
    training_context = _historical_frozen_context_profile(
        root,
        name="training-v1.1",
        manifest_relative_path=V1_1_MANIFEST_RELATIVE_PATH,
        expected_manifest_sha256=V1_1_MANIFEST_SHA256,
        source={"commit": V1_1_RESULT_SOURCE_COMMIT, "dirty": False},
    )
    context_profiles = [current_context, training_context]
    current_context_sha256 = cast(str, current_context["profile_sha256"])
    training_context_sha256 = cast(str, training_context["profile_sha256"])
    current_manifest_path = cast(str, current_context["manifest_path"])
    training_manifest_path = cast(str, training_context["manifest_path"])
    outer_profiles: list[dict[str, Any]] = []
    row_by_coordinate: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        coordinate = (row.get("scale"), row.get("training_seed"))
        _require(
            coordinate[0] in SCALES
            and coordinate[1] in TRAINING_SEEDS
            and coordinate not in row_by_coordinate,
            "Historical calibration-provenance coordinate is invalid.",
        )
        key = cast(tuple[str, int], coordinate)
        row_by_coordinate[key] = row
        profile = {
            "coordinate": {"scale": key[0], "training_seed": key[1]},
            "entry_cwd_mode": "detached",
            "checkpoint_path": row.get("checkpoint_relative_path"),
            "manifest_path": current_manifest_path,
            "training_manifest_path": training_manifest_path,
            "training_summary_path": str(
                Path(cast(str, row.get("resolved_path"))) / f"{key[0]}-training.summary.json"
            ),
            "training_matrix_summary_path": str(root / HISTORICAL_TRAINING_LEDGER),
            "optional_path_arguments": "all-explicit",
            "ordered_inner_context_profiles": [
                current_context_sha256,
                training_context_sha256,
            ],
        }
        outer_profiles.append(
            {
                "profile": profile,
                "profile_sha256": _json_digest(profile),
                "expected_invocations": 6,
            }
        )
    first = row_by_coordinate[(SCALES[0], TRAINING_SEEDS[0])]
    legacy_profile = {
        "coordinate": {"scale": SCALES[0], "training_seed": TRAINING_SEEDS[0]},
        "entry_cwd_mode": "canonical-root",
        "checkpoint_path": first.get("checkpoint_relative_path"),
        "manifest_path": training_manifest_path,
        "training_manifest_path": training_manifest_path,
        "training_summary_path": str(
            Path(cast(str, first.get("resolved_path"))) / f"{SCALES[0]}-training.summary.json"
        ),
        "training_matrix_summary_path": str(root / HISTORICAL_TRAINING_LEDGER),
        "optional_path_arguments": "all-explicit",
        "ordered_inner_context_profiles": [
            training_context_sha256,
            training_context_sha256,
        ],
    }
    outer_profiles.append(
        {
            "profile": legacy_profile,
            "profile_sha256": _json_digest(legacy_profile),
            "expected_invocations": 1,
        }
    )
    outer_profiles.sort(key=lambda row: cast(str, row["profile_sha256"]))
    ordered_pairs = [
        {
            "ordered_context_profile_sha256": [
                current_context_sha256,
                training_context_sha256,
            ],
            "expected_invocations": 60,
        },
        {
            "ordered_context_profile_sha256": [
                training_context_sha256,
                training_context_sha256,
            ],
            "expected_invocations": 1,
        },
    ]
    ordered_pairs.sort(key=lambda row: _json_digest(row["ordered_context_profile_sha256"]))
    context_invocations = [
        {"profile_sha256": current_context_sha256, "expected_invocations": 60},
        {"profile_sha256": training_context_sha256, "expected_invocations": 62},
    ]
    context_invocations.sort(key=lambda row: cast(str, row["profile_sha256"]))
    return {
        "schema_version": 1,
        "functions": {
            "outer": "calibration.establish_provenance",
            "inner": "calibration._frozen_context_for_manifest",
        },
        "cwd_translation": {
            "outer": "detached-or-authorized-root-to-canonical-root-to-same-entry",
            "inner": "authorized-root-to-detached-source-validation-to-authorized-root",
        },
        "callable_identity_restored": True,
        "outer_profiles": outer_profiles,
        "outer_profile_count": 11,
        "outer_invocation_count": 61,
        "ordered_inner_pairs": ordered_pairs,
        "ordered_inner_pair_count": 2,
        "inner_context_invocations": context_invocations,
        "inner_invocation_count": 122,
        "implicit_v1_2_source_state_invocation_count": 180,
        "outer_invocation_multiset_sha256": _json_digest(outer_profiles),
        "ordered_inner_pair_multiset_sha256": _json_digest(ordered_pairs),
        "inner_context_invocation_multiset_sha256": _json_digest(context_invocations),
        "context_profiles": context_profiles,
    }


def _verify_historical_calibration_provenance_claim(raw: object) -> None:
    _require(
        isinstance(raw, Mapping)
        and set(raw)
        == {
            "schema_version",
            "functions",
            "cwd_translation",
            "callable_identity_restored",
            "outer_profiles",
            "outer_profile_count",
            "outer_invocation_count",
            "ordered_inner_pairs",
            "ordered_inner_pair_count",
            "inner_context_invocations",
            "inner_invocation_count",
            "implicit_v1_2_source_state_invocation_count",
            "outer_invocation_multiset_sha256",
            "ordered_inner_pair_multiset_sha256",
            "inner_context_invocation_multiset_sha256",
            "context_profiles",
        }
        and raw.get("schema_version") == 1
        and raw.get("functions")
        == {
            "outer": "calibration.establish_provenance",
            "inner": "calibration._frozen_context_for_manifest",
        }
        and raw.get("cwd_translation")
        == {
            "outer": "detached-or-authorized-root-to-canonical-root-to-same-entry",
            "inner": "authorized-root-to-detached-source-validation-to-authorized-root",
        }
        and raw.get("callable_identity_restored") is True
        and raw.get("outer_profile_count") == 11
        and raw.get("outer_invocation_count") == 61
        and raw.get("ordered_inner_pair_count") == 2
        and raw.get("inner_invocation_count") == 122
        and raw.get("implicit_v1_2_source_state_invocation_count") == 180
        and _is_sha256(raw.get("outer_invocation_multiset_sha256"))
        and _is_sha256(raw.get("ordered_inner_pair_multiset_sha256"))
        and _is_sha256(raw.get("inner_context_invocation_multiset_sha256")),
        "Historical calibration-provenance adapter claim drifted.",
    )
    raw_mapping = cast(Mapping[str, Any], raw)
    raw_outer_profiles = raw_mapping.get("outer_profiles")
    raw_ordered_pairs = raw_mapping.get("ordered_inner_pairs")
    raw_inner_invocations = raw_mapping.get("inner_context_invocations")
    raw_context_profiles = raw_mapping.get("context_profiles")
    _require(
        isinstance(raw_outer_profiles, list)
        and isinstance(raw_ordered_pairs, list)
        and isinstance(raw_inner_invocations, list)
        and isinstance(raw_context_profiles, list),
        "Historical calibration-provenance profile inventory is malformed.",
    )
    outer_profiles = cast(list[Any], raw_outer_profiles)
    ordered_pairs = cast(list[Any], raw_ordered_pairs)
    inner_invocations = cast(list[Any], raw_inner_invocations)
    context_profiles = cast(list[Any], raw_context_profiles)
    _require(
        len(outer_profiles) == 11
        and all(
            isinstance(row, Mapping)
            and set(row) == {"profile", "profile_sha256", "expected_invocations"}
            and isinstance(row.get("profile"), Mapping)
            and row.get("profile_sha256") == _json_digest(row.get("profile"))
            and type(row.get("expected_invocations")) is int
            for row in outer_profiles
        )
        and sum(
            cast(int, cast(Mapping[str, Any], row)["expected_invocations"])
            for row in outer_profiles
        )
        == 61
        and len(ordered_pairs) == 2
        and all(
            isinstance(row, Mapping)
            and set(row) == {"ordered_context_profile_sha256", "expected_invocations"}
            and isinstance(row.get("ordered_context_profile_sha256"), list)
            and len(cast(list[Any], row.get("ordered_context_profile_sha256"))) == 2
            and all(
                _is_sha256(item)
                for item in cast(list[Any], row.get("ordered_context_profile_sha256"))
            )
            and type(row.get("expected_invocations")) is int
            for row in ordered_pairs
        )
        and sorted(
            cast(int, cast(Mapping[str, Any], row)["expected_invocations"]) for row in ordered_pairs
        )
        == [1, 60]
        and len(inner_invocations) == 2
        and all(
            isinstance(row, Mapping)
            and set(row) == {"profile_sha256", "expected_invocations"}
            and _is_sha256(row.get("profile_sha256"))
            and type(row.get("expected_invocations")) is int
            for row in inner_invocations
        )
        and sorted(
            cast(int, cast(Mapping[str, Any], row)["expected_invocations"])
            for row in inner_invocations
        )
        == [60, 62]
        and len(context_profiles) == 2
        and all(
            isinstance(row, Mapping)
            and set(row)
            == {"name", "manifest_path", "manifest_binding", "source", "profile_sha256"}
            and isinstance(row.get("manifest_binding"), Mapping)
            and isinstance(row.get("source"), Mapping)
            and row.get("profile_sha256")
            == _json_digest({key: value for key, value in row.items() if key != "profile_sha256"})
            for row in context_profiles
        )
        and raw_mapping.get("outer_invocation_multiset_sha256") == _json_digest(outer_profiles)
        and raw_mapping.get("ordered_inner_pair_multiset_sha256") == _json_digest(ordered_pairs)
        and raw_mapping.get("inner_context_invocation_multiset_sha256")
        == _json_digest(inner_invocations),
        "Historical calibration-provenance profile inventory drifted.",
    )
    exact_context_rows = [cast(Mapping[str, Any], row) for row in context_profiles]
    context_by_name = {cast(str, row.get("name")): row for row in exact_context_rows}
    _require(
        set(context_by_name) == {"current-v1.2", "training-v1.1"}
        and context_by_name["current-v1.2"].get("source")
        == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False}
        and context_by_name["training-v1.1"].get("source")
        == {"commit": V1_1_RESULT_SOURCE_COMMIT, "dirty": False}
        and all(
            isinstance(row.get("manifest_path"), str)
            and Path(cast(str, row.get("manifest_path"))).is_absolute()
            and _is_sha256(row.get("profile_sha256"))
            for row in exact_context_rows
        ),
        "Historical calibration-provenance frozen contexts drifted.",
    )
    current_context = context_by_name["current-v1.2"]
    training_context = context_by_name["training-v1.1"]
    current_sha256 = cast(str, current_context["profile_sha256"])
    training_sha256 = cast(str, training_context["profile_sha256"])
    current_manifest_path = cast(str, current_context["manifest_path"])
    training_manifest_path = cast(str, training_context["manifest_path"])
    exact_outer_rows = [cast(Mapping[str, Any], row) for row in outer_profiles]
    detached_outer_rows = [
        row
        for row in exact_outer_rows
        if cast(Mapping[str, Any], row["profile"]).get("entry_cwd_mode") == "detached"
    ]
    root_outer_rows = [
        row
        for row in exact_outer_rows
        if cast(Mapping[str, Any], row["profile"]).get("entry_cwd_mode") == "canonical-root"
    ]
    expected_coordinates = {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS}
    _require(
        len(detached_outer_rows) == 10
        and len(root_outer_rows) == 1
        and len({cast(str, row["profile_sha256"]) for row in exact_outer_rows}) == 11
        and {
            (
                cast(Mapping[str, Any], cast(Mapping[str, Any], row["profile"])["coordinate"])[
                    "scale"
                ],
                cast(Mapping[str, Any], cast(Mapping[str, Any], row["profile"])["coordinate"])[
                    "training_seed"
                ],
            )
            for row in detached_outer_rows
        }
        == expected_coordinates
        and all(
            row.get("expected_invocations") == 6
            and set(cast(Mapping[str, Any], row["profile"]))
            == {
                "coordinate",
                "entry_cwd_mode",
                "checkpoint_path",
                "manifest_path",
                "training_manifest_path",
                "training_summary_path",
                "training_matrix_summary_path",
                "optional_path_arguments",
                "ordered_inner_context_profiles",
            }
            and cast(Mapping[str, Any], row["profile"]).get("manifest_path")
            == current_manifest_path
            and cast(Mapping[str, Any], row["profile"]).get("training_manifest_path")
            == training_manifest_path
            and cast(Mapping[str, Any], row["profile"]).get("optional_path_arguments")
            == "all-explicit"
            and cast(Mapping[str, Any], row["profile"]).get("ordered_inner_context_profiles")
            == [current_sha256, training_sha256]
            and all(
                isinstance(cast(Mapping[str, Any], row["profile"]).get(field), str)
                for field in (
                    "checkpoint_path",
                    "training_summary_path",
                    "training_matrix_summary_path",
                )
            )
            for row in detached_outer_rows
        ),
        "Historical calibration-provenance current outer profiles drifted.",
    )
    legacy_row = root_outer_rows[0]
    legacy_profile = cast(Mapping[str, Any], legacy_row["profile"])
    _require(
        legacy_row.get("expected_invocations") == 1
        and set(legacy_profile)
        == {
            "coordinate",
            "entry_cwd_mode",
            "checkpoint_path",
            "manifest_path",
            "training_manifest_path",
            "training_summary_path",
            "training_matrix_summary_path",
            "optional_path_arguments",
            "ordered_inner_context_profiles",
        }
        and legacy_profile.get("coordinate")
        == {"scale": SCALES[0], "training_seed": TRAINING_SEEDS[0]}
        and legacy_profile.get("manifest_path") == training_manifest_path
        and legacy_profile.get("training_manifest_path") == training_manifest_path
        and legacy_profile.get("optional_path_arguments") == "all-explicit"
        and legacy_profile.get("ordered_inner_context_profiles")
        == [training_sha256, training_sha256]
        and all(
            isinstance(legacy_profile.get(field), str)
            for field in (
                "checkpoint_path",
                "training_summary_path",
                "training_matrix_summary_path",
            )
        ),
        "Historical calibration-provenance legacy outer profile drifted.",
    )
    exact_pair_rows = [cast(Mapping[str, Any], row) for row in ordered_pairs]
    exact_inner_rows = [cast(Mapping[str, Any], row) for row in inner_invocations]
    _require(
        {
            tuple(cast(list[str], row["ordered_context_profile_sha256"])): row[
                "expected_invocations"
            ]
            for row in exact_pair_rows
        }
        == {
            (current_sha256, training_sha256): 60,
            (training_sha256, training_sha256): 1,
        }
        and {row["profile_sha256"]: row["expected_invocations"] for row in exact_inner_rows}
        == {current_sha256: 60, training_sha256: 62},
        "Historical calibration-provenance inner multiset semantics drifted.",
    )


def _assert_historical_calibration_provenance_observation(
    claim: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> None:
    outer_expected = {
        cast(str, row["profile_sha256"]): cast(int, row["expected_invocations"])
        for row in cast(list[Mapping[str, Any]], claim["outer_profiles"])
    }
    ordered_expected = {
        _json_digest(row["ordered_context_profile_sha256"]): cast(int, row["expected_invocations"])
        for row in cast(list[Mapping[str, Any]], claim["ordered_inner_pairs"])
    }
    inner_expected = {
        cast(str, row["profile_sha256"]): cast(int, row["expected_invocations"])
        for row in cast(list[Mapping[str, Any]], claim["inner_context_invocations"])
    }
    _require(
        set(observation)
        == {
            "outer_profiles",
            "ordered_inner_pairs",
            "inner_context_profiles",
            "implicit_v1_2_source_state",
        }
        and observation.get("outer_profiles") == outer_expected
        and observation.get("ordered_inner_pairs") == ordered_expected
        and observation.get("inner_context_profiles") == inner_expected
        and observation.get("implicit_v1_2_source_state")
        == claim.get("implicit_v1_2_source_state_invocation_count"),
        "Historical calibration-provenance observed invocation multiset drifted.",
    )


@contextmanager
def _legacy_calibration_provenance_cwd_adapters(
    calibration: Any,
    *,
    repository_root: Path,
    detached_root: Path,
    trust_root: attestation.TrustRoot,
    claim: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    _verify_historical_calibration_provenance_claim(claim)
    raw_outer = getattr(calibration, "establish_provenance", None)
    raw_inner = getattr(calibration, "_frozen_context_for_manifest", None)
    training_matrix = getattr(calibration, "training_matrix", None)
    contract_module = getattr(training_matrix, "contract", None)
    raw_source_state = getattr(contract_module, "source_state", None)
    _require(
        callable(raw_outer)
        and getattr(raw_outer, "__name__", None) == "establish_provenance"
        and not bool(getattr(raw_outer, "_adaptive_v4_provenance_outer_adapter", False))
        and callable(raw_inner)
        and getattr(raw_inner, "__name__", None) == "_frozen_context_for_manifest"
        and not bool(getattr(raw_inner, "_adaptive_v4_provenance_inner_adapter", False))
        and callable(raw_source_state)
        and getattr(raw_source_state, "__name__", None) == "source_state",
        "Legacy calibration provenance callables are unavailable, aliased, or already adapted.",
    )
    outer_original = cast(Callable[..., Any], raw_outer)
    inner_original = cast(Callable[..., Any], raw_inner)
    source_state_original = cast(Callable[..., Any], raw_source_state)
    outer_rows = {
        cast(str, row["profile_sha256"]): row
        for row in cast(list[Mapping[str, Any]], claim["outer_profiles"])
    }
    context_rows = {
        cast(str, row["profile_sha256"]): row
        for row in cast(list[Mapping[str, Any]], claim["context_profiles"])
    }
    exact_contract_module = cast(Any, contract_module)
    context_by_path = {
        cast(str, row["manifest_path"]): profile_sha256
        for profile_sha256, row in context_rows.items()
    }
    observed_outer: dict[str, int] = {}
    observed_pairs: dict[str, int] = {}
    observed_inner: dict[str, int] = {}
    observed = {
        "outer_profiles": observed_outer,
        "ordered_inner_pairs": observed_pairs,
        "inner_context_profiles": observed_inner,
        "implicit_v1_2_source_state": 0,
    }

    def inner_adapter(*arguments: Any, **keywords: Any) -> Any:
        authority = _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get()
        _require(
            type(authority) is _HistoricalCalibrationProvenanceAuthority
            and authority.seal is _HISTORICAL_CALIBRATION_PROVENANCE_SEAL
            and _historical_thread_fs_capability_is_installed(authority.capability),
            "Legacy calibration inner provenance call lacks exact outer authority.",
        )
        exact_authority = cast(_HistoricalCalibrationProvenanceAuthority, authority)
        exact_authority.observed_inner_attempts.append(None)
        _require(
            len(exact_authority.observed_inner_attempts) <= 2
            and len(arguments) == 1
            and isinstance(arguments[0], Path)
            and set(keywords) == {"trust_root"}
            and keywords.get("trust_root") is trust_root
            and len(exact_authority.observed_inner_profiles) < 2
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "canonical-root",
            "Legacy calibration inner provenance call lacks exact outer authority.",
        )
        manifest_path = cast(Path, arguments[0])
        profile_sha256 = context_by_path.get(str(manifest_path))
        index = len(exact_authority.observed_inner_profiles)
        _require(
            profile_sha256 is not None
            and profile_sha256 == exact_authority.expected_inner_profiles[index]
            and exact_contract_module.source_state is source_state_original,
            "Legacy calibration frozen-context order or manifest path drifted.",
        )
        exact_profile_sha256 = cast(str, profile_sha256)
        context_row = context_rows[exact_profile_sha256]
        source_state_calls = 0
        expected_source_state_calls = 3 if context_row.get("name") == "current-v1.2" else 0

        def source_state_adapter(*source_arguments: Any, **source_keywords: Any) -> Any:
            nonlocal source_state_calls
            source_state_calls += 1
            _require(
                not source_arguments
                and not source_keywords
                and source_state_calls <= expected_source_state_calls
                and _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is exact_authority
                and _cwd_matches(
                    detached_root,
                    exact_authority.capability.detached_identity,
                ),
                "Historical source-state call escaped its exact detached provenance profile.",
            )
            state = source_state_original()
            _require(
                state == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
                "Historical implicit source-state result drifted.",
            )
            return state

        source_state_adapter.__name__ = "source_state"
        exact_contract_module.source_state = source_state_adapter
        try:
            with _scoped_historical_source_validation_cwd(
                repository_root=repository_root,
                detached_root=detached_root,
            ):
                result = inner_original(*arguments, **keywords)
        finally:
            source_identity_held = exact_contract_module.source_state is source_state_adapter
            exact_contract_module.source_state = source_state_original
            source_identity_held = (
                source_identity_held and exact_contract_module.source_state is source_state_original
            )
            _require(
                source_identity_held and source_state_calls == expected_source_state_calls,
                "Historical source-state invocation count or callable identity drifted.",
            )
        _require(
            getattr(result, "manifest_path", None) == Path(cast(str, context_row["manifest_path"]))
            and getattr(result, "manifest_binding", None) == context_row.get("manifest_binding")
            and getattr(result, "source", None) == context_row.get("source")
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "canonical-root",
            "Historical calibration frozen context or root restoration drifted.",
        )
        exact_authority.observed_inner_profiles.append(exact_profile_sha256)
        observed_inner[exact_profile_sha256] = observed_inner.get(exact_profile_sha256, 0) + 1
        observed["implicit_v1_2_source_state"] = (
            cast(int, observed["implicit_v1_2_source_state"]) + source_state_calls
        )
        return result

    def outer_adapter(*arguments: Any, **keywords: Any) -> Any:
        entry_mode = _historical_cwd_authority_mode(
            repository_root=repository_root,
            detached_root=detached_root,
        )
        expected_keys = {
            "scale",
            "training_seed",
            "manifest_path",
            "training_manifest_path",
            "training_summary_path",
            "training_matrix_summary_path",
            "trust_root",
        }
        _require(
            len(arguments) == 1
            and isinstance(arguments[0], Path)
            and set(keywords) == expected_keys
            and keywords.get("scale") in SCALES
            and keywords.get("training_seed") in TRAINING_SEEDS
            and all(
                isinstance(keywords.get(name), Path)
                for name in (
                    "manifest_path",
                    "training_manifest_path",
                    "training_summary_path",
                    "training_matrix_summary_path",
                )
            )
            and keywords.get("trust_root") is trust_root,
            "Legacy calibration provenance arguments are incomplete or invalid.",
        )
        profile = {
            "coordinate": {
                "scale": keywords["scale"],
                "training_seed": keywords["training_seed"],
            },
            "entry_cwd_mode": entry_mode,
            "checkpoint_path": str(arguments[0]),
            "manifest_path": str(keywords["manifest_path"]),
            "training_manifest_path": str(keywords["training_manifest_path"]),
            "training_summary_path": str(keywords["training_summary_path"]),
            "training_matrix_summary_path": str(keywords["training_matrix_summary_path"]),
            "optional_path_arguments": "all-explicit",
            "ordered_inner_context_profiles": [],
        }
        matching = [
            (profile_sha256, row)
            for profile_sha256, row in outer_rows.items()
            if {
                key: value
                for key, value in cast(Mapping[str, Any], row["profile"]).items()
                if key != "ordered_inner_context_profiles"
            }
            == {
                key: value
                for key, value in profile.items()
                if key != "ordered_inner_context_profiles"
            }
        ]
        _require(
            len(matching) == 1,
            "Legacy calibration provenance arguments differ from signed profiles.",
        )
        profile_sha256, profile_row = matching[0]
        expected_inner = tuple(
            cast(Mapping[str, Any], profile_row["profile"])["ordered_inner_context_profiles"]
        )
        _require(
            len(expected_inner) == 2
            and all(_is_sha256(item) for item in expected_inner)
            and _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is None,
            "Legacy calibration provenance inner profile authority is invalid.",
        )
        capability = _get_installed_historical_thread_fs_capability()
        _require(
            capability is not None and _historical_thread_fs_capability_is_installed(capability),
            "Legacy calibration provenance lacks installed cwd isolation.",
        )
        authority = _HistoricalCalibrationProvenanceAuthority(
            seal=_HISTORICAL_CALIBRATION_PROVENANCE_SEAL,
            capability=cast(_HistoricalThreadFsIsolationCapability, capability),
            outer_profile_sha256=profile_sha256,
            expected_inner_profiles=cast(tuple[str, str], expected_inner),
            observed_inner_attempts=[],
            observed_inner_profiles=[],
        )
        with _scoped_historical_command_resolution_cwd(
            repository_root=repository_root,
            detached_root=detached_root,
        ):
            provenance_token = _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.set(authority)
            cleanup_errors: list[BaseException] = []
            original_error: BaseException | None = None
            try:
                result = outer_original(*arguments, **keywords)
            except BaseException as error:
                original_error = error
                raise
            finally:
                authority_held = _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is authority
                inner_complete = len(authority.observed_inner_attempts) == 2 and tuple(
                    authority.observed_inner_profiles
                ) == cast(tuple[str, str], expected_inner)
                try:
                    _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.reset(provenance_token)
                except BaseException as error:
                    cleanup_errors.append(error)
                if _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is not None:
                    try:
                        _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.set(None)
                    except BaseException as error:
                        cleanup_errors.append(error)
                authority_cleared = _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is None
                if not (authority_held and inner_complete and authority_cleared):
                    cleanup_errors.append(
                        ValueError("Legacy calibration provenance authority or order drifted.")
                    )
                if cleanup_errors:
                    raise ValueError("Legacy calibration provenance cleanup failed closed.") from (
                        original_error or cleanup_errors[0]
                    )
        _require(
            _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == entry_mode
            and _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is None,
            "Legacy calibration provenance did not restore its entry cwd authority.",
        )
        observed_outer[profile_sha256] = observed_outer.get(profile_sha256, 0) + 1
        pair_sha256 = _json_digest(list(expected_inner))
        observed_pairs[pair_sha256] = observed_pairs.get(pair_sha256, 0) + 1
        return result

    outer_adapter.__name__ = "establish_provenance"
    inner_adapter.__name__ = "_frozen_context_for_manifest"
    outer_adapter._adaptive_v4_provenance_outer_adapter = True  # type: ignore[attr-defined]
    inner_adapter._adaptive_v4_provenance_inner_adapter = True  # type: ignore[attr-defined]
    calibration.establish_provenance = outer_adapter
    calibration._frozen_context_for_manifest = inner_adapter
    try:
        yield observed
    finally:
        identities_held = (
            calibration.establish_provenance is outer_adapter
            and calibration._frozen_context_for_manifest is inner_adapter
            and exact_contract_module.source_state is source_state_original
        )
        calibration._frozen_context_for_manifest = inner_original
        calibration.establish_provenance = outer_original
        _require(
            identities_held
            and calibration.establish_provenance is outer_original
            and calibration._frozen_context_for_manifest is inner_original
            and exact_contract_module.source_state is source_state_original
            and _HISTORICAL_CALIBRATION_PROVENANCE_AUTHORITY.get() is None
            and _HISTORICAL_SUPERSEDED_PATH_SPELLING_AUTHORITY.get() is None
            and _historical_cwd_authority_mode(
                repository_root=repository_root,
                detached_root=detached_root,
            )
            == "detached",
            "Legacy calibration provenance callable identity or cwd authority drifted.",
        )


@contextmanager
def _legacy_quarantine_cwd_adapter(
    calibration_matrix: Any,
    *,
    repository_root: Path,
    detached_root: Path,
    trust_root: attestation.TrustRoot,
    argument_claim: Mapping[str, Any],
) -> Iterator[dict[str, int]]:
    raw_original = getattr(calibration_matrix, "_validate_quarantine_evidence", None)
    _require(
        callable(raw_original)
        and getattr(raw_original, "__name__", None) == "_validate_quarantine_evidence"
        and not bool(getattr(raw_original, "_adaptive_v4_quarantine_cwd_adapter", False)),
        "Legacy quarantine validator is unavailable, aliased, or already adapted.",
    )
    original = cast(Callable[..., Any], raw_original)
    expected_keys = {
        "legacy_context",
        "trust_root",
        "training_matrix_summary_path",
        "training_matrix_payload",
        "trainer_binding",
        "ledger_records",
    }
    invocations: dict[str, int] = {}

    def adapter(*arguments: Any, **keywords: Any) -> Any:
        legacy_context = keywords.get("legacy_context")
        training_payload = keywords.get("training_matrix_payload")
        ledger_records = keywords.get("ledger_records")
        _require(
            not arguments
            and set(keywords) == expected_keys
            and keywords.get("trust_root") is trust_root
            and keywords.get("training_matrix_summary_path")
            == Path(cast(str, argument_claim.get("training_matrix_summary_path")))
            and isinstance(training_payload, Mapping)
            and _json_digest(training_payload)
            == argument_claim.get("training_matrix_payload_sha256")
            and keywords.get("trainer_binding") == argument_claim.get("trainer_binding")
            and isinstance(ledger_records, Mapping)
            and _json_digest(
                {
                    "schema_version": 1,
                    "records": _historical_ledger_record_inventory(
                        cast(Mapping[tuple[str, int], Mapping[str, Any]], ledger_records)
                    ),
                }
            )
            == argument_claim.get("ledger_record_inventory_sha256")
            and getattr(legacy_context, "source", None) == argument_claim.get("legacy_source")
            and getattr(legacy_context, "manifest_binding", None)
            == argument_claim.get("legacy_manifest")
            and getattr(legacy_context, "manifest_path", None)
            == Path(cast(str, cast(Mapping[str, Any], argument_claim["legacy_manifest"])["path"]))
            and threading.current_thread() is threading.main_thread()
            and threading.active_count() == 1
            and Path.cwd() == detached_root
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy quarantine adapter arguments, thread, or detached source drifted.",
        )
        profile_sha256 = cast(str, argument_claim.get("expected_invocation_profile_sha256"))
        _require(_is_sha256(profile_sha256), "Legacy quarantine invocation profile is invalid.")
        with _scoped_historical_command_resolution_cwd(
            repository_root=repository_root,
            detached_root=detached_root,
        ):
            result = original(**keywords)
        _require(
            Path.cwd() == detached_root
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy quarantine adapter did not restore the detached source.",
        )
        invocations[profile_sha256] = invocations.get(profile_sha256, 0) + 1
        return result

    adapter._adaptive_v4_quarantine_cwd_adapter = True  # type: ignore[attr-defined]
    calibration_matrix._validate_quarantine_evidence = adapter
    try:
        yield invocations
    finally:
        adapter_held = getattr(calibration_matrix, "_validate_quarantine_evidence", None) is adapter
        calibration_matrix._validate_quarantine_evidence = original
        _require(
            adapter_held
            and getattr(calibration_matrix, "_validate_quarantine_evidence", None) is original
            and Path.cwd() == detached_root
            and _source_state(detached_root)
            == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
            "Legacy quarantine callable identity or detached source drifted.",
        )


def _archived_validation_payload(
    *,
    trust_root: attestation.TrustRoot,
    repository_root: Path,
) -> dict[str, Any]:
    # Imported here so importing this admission module cannot materialize quality coordinates,
    # touch CUDA, or import either the old or new controller evaluator.
    import run_p2_direct_top_p_physical_matrix as top_p_matrix

    detached_root = _exact_path(
        Path.cwd(),
        label="Detached historical result-source root",
        must_exist=True,
    )
    detached_state = _source_state(detached_root)
    _require(
        detached_state == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
        "Archived child is not running in the exact detached result source.",
    )
    _assert_tree_object(
        detached_root,
        HISTORICAL_RESULT_SOURCE_COMMIT,
        HISTORICAL_RESULT_SOURCE_TREE,
    )
    _assert_tree_object(
        detached_root,
        HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
        HISTORICAL_IMPLEMENTATION_SOURCE_TREE,
    )
    _assert_tree_object(detached_root, V1_1_RESULT_SOURCE_COMMIT, V1_1_RESULT_SOURCE_TREE)
    _assert_tree_object(
        detached_root,
        V1_1_IMPLEMENTATION_SOURCE_COMMIT,
        V1_1_IMPLEMENTATION_SOURCE_TREE,
    )
    _assert_ancestor(
        detached_root,
        HISTORICAL_IMPLEMENTATION_SOURCE_COMMIT,
        HISTORICAL_RESULT_SOURCE_COMMIT,
    )
    _assert_ancestor(
        detached_root,
        V1_1_IMPLEMENTATION_SOURCE_COMMIT,
        V1_1_RESULT_SOURCE_COMMIT,
    )
    _assert_ancestor(detached_root, V1_1_RESULT_SOURCE_COMMIT, HISTORICAL_RESULT_SOURCE_COMMIT)
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

    runtime_binding = _archived_python_runtime_binding(repository_root)
    command_path_inventory = _historical_training_command_path_inventory(repository_root)
    builder_inventories = _historical_signed_builder_command_inventory(
        repository_root,
        command_path_inventory,
    )
    calibration_provenance_claim = _historical_calibration_provenance_claim(
        repository_root,
        command_path_inventory,
    )
    _require(
        top_p_matrix.training_matrix is top_p_matrix.calibration_matrix.training_matrix
        and top_p_matrix.calibration_program is top_p_matrix.calibration_matrix.calibration,
        "Archived prerequisite modules do not share the exact legacy training validator.",
    )
    expected_training_coordinates = {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS}
    _require(
        Path.cwd() == detached_root
        and top_p_matrix.contract.source_state()
        == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
        "Archived implicit source state drifted before prerequisite validation.",
    )
    superseded_argument_claim = _historical_superseded_bundle_argument_claim(repository_root)
    quarantine_argument_claim = _historical_quarantine_argument_claim(repository_root)
    retry_admission_cwd_claim = _historical_retry_admission_cwd_claim(repository_root)
    thread_fs_isolation_claim = _unshare_validating_thread_fs_context(
        repository_root=repository_root,
        detached_root=detached_root,
    )
    with _retained_interpreter_spelling(repository_root, runtime_binding) as retained_interpreter:
        builder_specs = (
            (
                top_p_matrix.training_matrix,
                "build_training_command",
                "training_matrix.build_training_command",
            ),
            (
                top_p_matrix.calibration_matrix,
                "build_calibration_command",
                "calibration_matrix.build_calibration_command",
            ),
            (
                top_p_matrix,
                "build_generator_command",
                "top_p_matrix.build_generator_command",
            ),
        )
        with ExitStack() as adapter_stack:
            validated_builder_commands = adapter_stack.enter_context(
                _legacy_builder_spelling_adapters(
                    builder_specs,
                    repository_root=repository_root,
                    detached_root=detached_root,
                    retained=retained_interpreter,
                    inventories=builder_inventories,
                )
            )
            superseded_invocations = adapter_stack.enter_context(
                _legacy_superseded_bundle_cwd_adapter(
                    top_p_matrix.training_matrix,
                    repository_root=repository_root,
                    detached_root=detached_root,
                    trust_root=trust_root,
                    argument_claim=superseded_argument_claim,
                )
            )
            validated_training_paths = adapter_stack.enter_context(
                _legacy_training_command_cwd_adapter(
                    top_p_matrix.training_matrix,
                    repository_root=repository_root,
                    detached_root=detached_root,
                    inventory=command_path_inventory,
                )
            )
            validated_calibration_provenance = adapter_stack.enter_context(
                _legacy_calibration_provenance_cwd_adapters(
                    top_p_matrix.calibration_program,
                    repository_root=repository_root,
                    detached_root=detached_root,
                    trust_root=trust_root,
                    claim=calibration_provenance_claim,
                )
            )
            adapter_stack.enter_context(
                _legacy_retry_admission_cwd_adapter(
                    top_p_matrix.calibration_matrix,
                    repository_root=repository_root,
                    detached_root=detached_root,
                    trust_root=trust_root,
                    claim=retry_admission_cwd_claim,
                )
            )
            quarantine_invocations = adapter_stack.enter_context(
                _legacy_quarantine_cwd_adapter(
                    top_p_matrix.calibration_matrix,
                    repository_root=repository_root,
                    detached_root=detached_root,
                    trust_root=trust_root,
                    argument_claim=quarantine_argument_claim,
                )
            )
            prerequisites = top_p_matrix.load_and_validate_prerequisites(
                manifest_path=repository_root / HISTORICAL_MANIFEST_RELATIVE_PATH,
                training_output_root=repository_root / HISTORICAL_TRAINING_ROOT,
                calibration_output_root=repository_root / HISTORICAL_CALIBRATION_ROOT,
                output_root=repository_root / HISTORICAL_TOP_P_ROOT,
                attestation_key_path=None,
            )
            _require(
                Path.cwd() == detached_root
                and top_p_matrix.contract.source_state()
                == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False}
                and _source_state(detached_root)
                == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
                "Archived implicit source state or detached cwd drifted after command validation.",
            )
            _require(
                prerequisites.context.source
                == {"commit": HISTORICAL_RESULT_SOURCE_COMMIT, "dirty": False},
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
            expected_training_path_counts = {
                coordinate: (71 if coordinate == (SCALES[0], TRAINING_SEEDS[0]) else 69)
                for coordinate in expected_training_coordinates
            }
            _require(
                validated_training_paths.get("training_command") == expected_training_path_counts
                and validated_training_paths.get("training_checkpoint")
                == expected_training_path_counts,
                "Archived legacy path validation invocation multiplicities drifted.",
            )
            relative_path_adapter_claim = _historical_relative_path_adapter_claim(
                command_path_inventory,
                superseded_argument_claim,
                quarantine_argument_claim,
                observed_training=validated_training_paths,
                observed_superseded=superseded_invocations,
                observed_quarantine=quarantine_invocations,
            )
            builder_adapter_claim = _historical_builder_adapter_claim(
                retained_interpreter,
                builder_inventories,
                observed_invocations=validated_builder_commands,
            )
            _assert_historical_calibration_provenance_observation(
                calibration_provenance_claim,
                validated_calibration_provenance,
            )
    command_path_resolution = _historical_command_path_resolution_claim(command_path_inventory)

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
        detached_root=detached_root,
        allowed_inventory=_commit_inventory(
            repository_root,
            HISTORICAL_RESULT_SOURCE_COMMIT,
            HISTORICAL_IMPLEMENTATION_PATHS,
        ),
        sealed_admission=_sealed_admission_module_binding(),
    )
    thread_fs_capability = _assert_thread_fs_isolation_capability(
        repository_root=repository_root,
        detached_root=detached_root,
        main_location="detached",
    )
    _require(
        canonical_json(thread_fs_isolation_claim) == thread_fs_capability.claim_bytes,
        "Historical thread fs-isolation receipt claim drifted from its private capability.",
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
            "historical_artifact_command_path_resolution": command_path_resolution,
            "historical_artifact_command_builder_adapter": builder_adapter_claim,
            "historical_calibration_provenance_cwd_adapter": calibration_provenance_claim,
            "historical_quarantine_cwd_adapter": quarantine_argument_claim,
            "historical_retry_admission_cwd_adapter": retry_admission_cwd_claim,
            "historical_relative_path_adapter_invocations": relative_path_adapter_claim,
            "historical_thread_fs_isolation": thread_fs_isolation_claim,
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
        "historical_artifact_command_path_resolution",
        "historical_artifact_command_builder_adapter",
        "historical_calibration_provenance_cwd_adapter",
        "historical_quarantine_cwd_adapter",
        "historical_retry_admission_cwd_adapter",
        "historical_relative_path_adapter_invocations",
        "historical_thread_fs_isolation",
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
    command_path_resolution = payload.get("historical_artifact_command_path_resolution")
    builder_adapter = payload.get("historical_artifact_command_builder_adapter")
    _verify_historical_calibration_provenance_claim(
        payload.get("historical_calibration_provenance_cwd_adapter")
    )
    _verify_historical_quarantine_argument_claim(payload.get("historical_quarantine_cwd_adapter"))
    _verify_historical_retry_admission_cwd_claim(
        payload.get("historical_retry_admission_cwd_adapter")
    )
    _verify_historical_relative_path_adapter_claim(
        payload.get("historical_relative_path_adapter_invocations")
    )
    _verify_historical_thread_fs_isolation_claim(payload.get("historical_thread_fs_isolation"))
    _verify_module_origin_audit(payload.get("module_origin_audit"))
    _require(
        isinstance(command_path_resolution, Mapping)
        and set(command_path_resolution)
        == {
            "schema_version",
            "adapter_scopes",
            "relative_path_fields",
            "relative_path_base",
            "source_and_import_validation_cwd",
            "cwd_switch",
            "cwd_restoration",
            "path_values_rewritten",
            "parent_traversal_allowed",
            "symlink_traversal_allowed",
            "validated_relative_path_count",
            "validated_relative_path_inventory_sha256",
        }
        and command_path_resolution.get("schema_version") == 1
        and command_path_resolution.get("adapter_scopes")
        == [
            "legacy-training-matrix._validate_training_command-only",
            "legacy-training-matrix._validate_checkpoint-only",
        ]
        and isinstance(command_path_resolution.get("relative_path_fields"), list)
        and [
            (row.get("field"), row.get("validated_count"))
            for row in cast(list[Any], command_path_resolution.get("relative_path_fields"))
            if isinstance(row, Mapping)
        ]
        == [
            ("training-summary.command.--output-dir", 10),
            ("training-summary.checkpoint.path", 10),
        ]
        and all(
            isinstance(row, Mapping)
            and set(row) == {"field", "validated_count", "inventory_sha256"}
            and _is_sha256(row.get("inventory_sha256"))
            for row in cast(list[Any], command_path_resolution.get("relative_path_fields"))
        )
        and command_path_resolution.get("relative_path_base")
        == "canonical-original-repository-root"
        and command_path_resolution.get("source_and_import_validation_cwd")
        == "detached-historical-result-source"
        and command_path_resolution.get("cwd_switch")
        == "exact-directory-fd-to-canonical-repository-root"
        and command_path_resolution.get("cwd_restoration")
        == "exact-directory-fd-to-detached-result-source"
        and command_path_resolution.get("path_values_rewritten") is False
        and command_path_resolution.get("parent_traversal_allowed") is False
        and command_path_resolution.get("symlink_traversal_allowed") is False
        and command_path_resolution.get("validated_relative_path_count")
        == 2 * len(SCALES) * len(TRAINING_SEEDS)
        and _is_sha256(command_path_resolution.get("validated_relative_path_inventory_sha256")),
        "Historical artifact command path-resolution claim drifted.",
    )
    _require(
        isinstance(builder_adapter, Mapping)
        and set(builder_adapter)
        == {
            "schema_version",
            "actual_child_process_executable_unchanged",
            "sys_executable_mutated",
            "reexec_performed",
            "adapter_operation",
            "nonzero_argv_bytes_and_order_preserved",
            "builder_replay_policy",
            "adapter_context_install_and_restore_cwd",
            "callable_identity_restored",
            "interpreter_spelling",
            "builders",
            "signed_command_count",
            "expected_builder_invocation_count",
        }
        and builder_adapter.get("schema_version") == 1
        and builder_adapter.get("actual_child_process_executable_unchanged") is True
        and builder_adapter.get("sys_executable_mutated") is False
        and builder_adapter.get("reexec_performed") is False
        and builder_adapter.get("adapter_operation")
        == "clone-original-builder-result-and-replace-index-zero-only"
        and builder_adapter.get("nonzero_argv_bytes_and_order_preserved") is True
        and builder_adapter.get("builder_replay_policy")
        == "exact-signed-membership-complete-coverage-repeats-allowed"
        and builder_adapter.get("adapter_context_install_and_restore_cwd")
        == "detached-result-source"
        and builder_adapter.get("callable_identity_restored") is True
        and builder_adapter.get("signed_command_count") == 60
        and builder_adapter.get("expected_builder_invocation_count") == 742,
        "Historical artifact command builder-adapter claim drifted.",
    )
    interpreter_spelling = cast(Mapping[str, Any], builder_adapter).get("interpreter_spelling")
    builders = cast(Mapping[str, Any], builder_adapter).get("builders")
    _require(
        isinstance(interpreter_spelling, Mapping)
        and set(interpreter_spelling)
        == {
            "schema_version",
            "recorded_executable",
            "alias_symlink_chain",
            "resolved_target",
            "target_metadata",
            "target_sha256",
        }
        and interpreter_spelling.get("schema_version") == 1
        and isinstance(interpreter_spelling.get("recorded_executable"), str)
        and Path(cast(str, interpreter_spelling.get("recorded_executable"))).is_absolute()
        and isinstance(interpreter_spelling.get("resolved_target"), str)
        and Path(cast(str, interpreter_spelling.get("resolved_target"))).is_absolute()
        and isinstance(interpreter_spelling.get("alias_symlink_chain"), list)
        and bool(interpreter_spelling.get("alias_symlink_chain"))
        and isinstance(interpreter_spelling.get("target_metadata"), Mapping)
        and _is_sha256(interpreter_spelling.get("target_sha256"))
        and isinstance(builders, list)
        and len(builders) == 3
        and all(isinstance(row, Mapping) for row in builders),
        "Historical interpreter spelling or legacy builder inventory claim drifted.",
    )
    builder_rows = [cast(Mapping[str, Any], row) for row in cast(list[Any], builders)]
    _require(
        [(row.get("builder"), row.get("signed_command_count")) for row in builder_rows]
        == [
            ("calibration_matrix.build_calibration_command", 10),
            ("top_p_matrix.build_generator_command", 40),
            ("training_matrix.build_training_command", 10),
        ]
        and all(
            set(row)
            == {
                "builder",
                "invocation_entry_cwd",
                "original_call_cwd",
                "invocation_exit_cwd",
                "signed_command_count",
                "signed_command_inventory_sha256",
                "expected_invocation_count",
                "expected_invocation_multiset_sha256",
            }
            and _is_sha256(row.get("signed_command_inventory_sha256"))
            and _is_sha256(row.get("expected_invocation_multiset_sha256"))
            for row in builder_rows
        )
        and [row.get("expected_invocation_count") for row in builder_rows] == [10, 40, 692],
        "Historical legacy builder command inventory claim drifted.",
    )
    _require(
        [
            (
                row.get("invocation_entry_cwd"),
                row.get("original_call_cwd"),
                row.get("invocation_exit_cwd"),
            )
            for row in builder_rows
        ]
        == [
            (
                "detached-result-source",
                "canonical-repository-root",
                "detached-result-source",
            ),
            (
                "detached-result-source",
                "detached-result-source",
                "detached-result-source",
            ),
            (
                "canonical-repository-root",
                "canonical-repository-root",
                "canonical-repository-root",
            ),
        ],
        "Historical legacy builder original-call cwd policy drifted.",
    )
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
    current_training_commands = _historical_training_command_path_inventory(
        quality_context.repository_root
    )
    current_command_path_resolution = _historical_command_path_resolution_claim(
        current_training_commands
    )
    _require(
        current_command_path_resolution
        == receipt.get("historical_artifact_command_path_resolution"),
        "Historical artifact command path-resolution evidence differs from the signed receipt.",
    )
    current_builder_inventories = _historical_signed_builder_command_inventory(
        quality_context.repository_root,
        current_training_commands,
    )
    current_runtime_binding = _archived_python_runtime_binding(quality_context.repository_root)
    with _retained_interpreter_spelling(
        quality_context.repository_root,
        current_runtime_binding,
    ) as retained_interpreter:
        current_builder_adapter = _historical_builder_adapter_claim(
            retained_interpreter,
            current_builder_inventories,
        )
    _require(
        current_builder_adapter == receipt.get("historical_artifact_command_builder_adapter"),
        "Historical command builder-adapter evidence differs from the signed receipt.",
    )
    current_calibration_provenance = _historical_calibration_provenance_claim(
        quality_context.repository_root,
        current_training_commands,
    )
    _require(
        current_calibration_provenance
        == receipt.get("historical_calibration_provenance_cwd_adapter"),
        "Historical calibration provenance evidence differs from the signed receipt.",
    )
    current_quarantine_claim = _historical_quarantine_argument_claim(
        quality_context.repository_root
    )
    _require(
        current_quarantine_claim == receipt.get("historical_quarantine_cwd_adapter"),
        "Historical quarantine cwd-adapter evidence differs from the signed receipt.",
    )
    current_retry_admission_claim = _historical_retry_admission_cwd_claim(
        quality_context.repository_root
    )
    _require(
        current_retry_admission_claim == receipt.get("historical_retry_admission_cwd_adapter"),
        "Historical retry-admission cwd evidence differs from the signed receipt.",
    )
    current_superseded_claim = _historical_superseded_bundle_argument_claim(
        quality_context.repository_root
    )
    _require(
        _historical_relative_path_adapter_claim(
            current_training_commands,
            current_superseded_claim,
            current_quarantine_claim,
        )
        == receipt.get("historical_relative_path_adapter_invocations"),
        "Historical relative-path invocation evidence differs from the signed receipt.",
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
        try:
            if staging is not None and not published and os.path.lexists(staging):
                _remove_safe_staging_directory(staging)
                _fsync_directory(parent)
        finally:
            lease.close()


def _exact_identity(path: Path, *, directory: bool) -> dict[str, Any]:
    metadata = os.stat(path, follow_symlinks=False)
    expected_mode = SAFE_DIRECTORY_MODE if directory else SAFE_FILE_MODE
    _require(
        (stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode))
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) == expected_mode
        and (directory or metadata.st_nlink == 1),
        f"Unsafe exact identity metadata: {path}",
    )
    return {
        "path": str(path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "mode": expected_mode,
        "nlink": metadata.st_nlink,
        "persistent_inode": True,
    }


def _assert_v1_3_2_quality_paths_absent(*, repository_root: Path) -> tuple[str, ...]:
    paths = tuple(
        _absolute(path, repository_root=repository_root)
        for path in V1_3_2_PROSPECTIVE_QUALITY_PATHS
    )
    for path in paths:
        _require(
            not os.path.lexists(path),
            f"Prospective v1.3.2 quality path exists before activation: {path}",
        )
    return tuple(str(path) for path in paths)


def _assert_v1_3_2_activation_root_absent(*, repository_root: Path) -> str:
    root = _absolute(V1_3_2_ACTIVATION_ROOT, repository_root=repository_root)
    _require(
        not os.path.lexists(root),
        f"V1.3.1 activation root already exists: {root}",
    )
    return str(root)


def _superseded_empty_quality_state() -> dict[str, Any]:
    return {
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
    }


def _superseded_manifest_binding(
    *, repository_root: Path, trust_root: attestation.TrustRoot
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _absolute(
        Path(
            "research/adaptive_v4_memory/manifests/"
            "p2-post-rank-direct-controller-exact-fill-v1-3.json"
        ),
        repository_root=repository_root,
    )
    payload, opened = _load_json_nofollow(
        path,
        label="Superseded v1.3 manifest",
        require_canonical_pretty_bytes=True,
    )
    try:
        implementation = payload.get("implementation")
        public_attestation = payload.get("attestation")
        expected_attestation = attestation.public_manifest_contract(trust_root.key_id)
        _require(
            opened.sha256 == V1_3_SUPERSEDED_MANIFEST_SHA256
            and opened.bytes == V1_3_SUPERSEDED_MANIFEST_BYTES
            and payload.get("schema_version") == 1
            and payload.get("experiment_id") == QUALITY_EXPERIMENT_ID
            and payload.get("status")
            == "frozen_after_v1_2_top_p_feasibility_no_go_before_any_held_out_quality"
            and isinstance(implementation, Mapping)
            and cast(Mapping[str, Any], implementation).get("source_commit")
            == V1_3_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT
            and cast(Mapping[str, Any], implementation).get("tree_digest")
            == V1_3_SUPERSEDED_IMPLEMENTATION_DIGEST
            and public_attestation == expected_attestation,
            "Superseded v1.3 manifest binding drifted.",
        )
        binding = {
            "path": str(path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
            "experiment_id": QUALITY_EXPERIMENT_ID,
            "implementation_source_commit": V1_3_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT,
            "implementation_digest": V1_3_SUPERSEDED_IMPLEMENTATION_DIGEST,
            "live_implementation_inventory_digest": (
                V1_3_SUPERSEDED_LIVE_IMPLEMENTATION_INVENTORY_DIGEST
            ),
            "live_implementation_file_count": V1_3_SUPERSEDED_LIVE_IMPLEMENTATION_FILE_COUNT,
            "attestation": expected_attestation,
        }
        opened.assert_unchanged()
        return dict(payload), binding
    finally:
        opened.close()


def load_superseded_empty_lineage_v1_3(
    *,
    trust_root: attestation.TrustRoot,
    repository_root: Path = REPOSITORY_ROOT,
) -> SupersededEmptyLineageV1_3:
    """Authenticate the one permitted v1.3 pair and prove its prefix is still empty."""

    _require(
        type(trust_root) is attestation.TrustRoot
        and trust_root.key_id == V1_3_SUPERSEDED_ATTESTATION_KEY_ID,
        "Superseded v1.3 lineage requires its exact historical trust root.",
    )
    root = _exact_path(repository_root, label="Repository root", must_exist=True)
    _assert_ancestor(
        root,
        V1_3_SUPERSEDED_RESULT_SOURCE_COMMIT,
        cast(str, _source_state(root)["commit"]),
    )
    _assert_tree_object(
        root,
        V1_3_SUPERSEDED_RESULT_SOURCE_COMMIT,
        V1_3_SUPERSEDED_RESULT_SOURCE_TREE,
    )
    _assert_tree_object(
        root,
        V1_3_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT,
        V1_3_SUPERSEDED_IMPLEMENTATION_SOURCE_TREE,
    )
    manifest_payload, manifest_binding = _superseded_manifest_binding(
        repository_root=root,
        trust_root=trust_root,
    )
    admission_root = _absolute(ADMISSION_ROOT, repository_root=root)
    admission_path, genesis_path = _validate_admission_bundle_root(
        admission_root,
        label="Superseded v1.3 admission root",
    )
    admission_payload, admission_opened = _load_json_nofollow(
        admission_path,
        label="Superseded v1.3 reuse admission",
        require_canonical_pretty_bytes=True,
    )
    try:
        _verify_attested_payload(
            admission_payload,
            trust_root=trust_root,
            purpose=REUSE_ADMISSION_PURPOSE,
            label="Superseded v1.3 reuse admission",
        )
        admission_binding = _reuse_admission_public_binding(
            path=admission_path,
            payload=admission_payload,
            sha256=admission_opened.sha256,
            byte_count=admission_opened.bytes,
        )
        envelope = admission_payload.get("attestation")
        receipt = admission_payload.get("historical_validation_receipt")
        nonobservation = admission_payload.get("canonical_nonobservation")
        _require(
            admission_opened.sha256 == V1_3_SUPERSEDED_ADMISSION_SHA256
            and admission_opened.bytes == V1_3_SUPERSEDED_ADMISSION_BYTES
            and admission_payload.get("payload_sha256") == V1_3_SUPERSEDED_ADMISSION_PAYLOAD_SHA256
            and isinstance(envelope, Mapping)
            and dict(cast(Mapping[str, Any], envelope))
            == {
                "scheme": "hmac-sha256-v1",
                "purpose": REUSE_ADMISSION_PURPOSE,
                "key_id": trust_root.key_id,
                "payload_sha256": V1_3_SUPERSEDED_ADMISSION_ATTESTATION_PAYLOAD_SHA256,
                "mac": V1_3_SUPERSEDED_ADMISSION_ATTESTATION_MAC,
            }
            and admission_payload.get("experiment_id") == QUALITY_EXPERIMENT_ID
            and admission_payload.get("canonical_path") == str(admission_path)
            and admission_payload.get("quality_source")
            == {"commit": V1_3_SUPERSEDED_RESULT_SOURCE_COMMIT, "dirty": False}
            and admission_payload.get("quality_manifest") == manifest_binding
            and isinstance(receipt, Mapping)
            and cast(Mapping[str, Any], receipt).get("payload_sha256")
            == V1_3_SUPERSEDED_HISTORICAL_RECEIPT_PAYLOAD_SHA256
            and isinstance(nonobservation, Mapping)
            and cast(Mapping[str, Any], nonobservation).get("payload_sha256")
            == V1_3_SUPERSEDED_CANONICAL_NONOBSERVATION_PAYLOAD_SHA256
            and admission_payload.get("quality_evaluation_started") is False
            and admission_payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
            and admission_payload.get("quality_rng_initialized") is False
            and all(
                admission_payload.get(field) == 0
                for field in (
                    "evaluation_inputs_materialized",
                    "quality_predictions_materialized",
                    "quality_outcomes_materialized",
                    "quality_aggregates_materialized",
                    "top_p_quality_input_count",
                    "scientific_subprocesses_started_during_admission",
                )
            ),
            "Superseded v1.3 reuse admission is not the exact signed empty lineage.",
        )
        admission_opened.assert_unchanged()
    finally:
        admission_opened.close()
    genesis_payload, genesis_opened = _load_json_nofollow(
        genesis_path,
        label="Superseded v1.3 pre-heldout genesis",
        require_canonical_pretty_bytes=True,
    )
    try:
        _verify_attested_payload(
            genesis_payload,
            trust_root=trust_root,
            purpose=PREHELDOUT_GENESIS_PURPOSE,
            label="Superseded v1.3 pre-heldout genesis",
        )
        genesis_binding = _genesis_public_binding(
            path=genesis_path,
            payload=genesis_payload,
            sha256=genesis_opened.sha256,
            byte_count=genesis_opened.bytes,
        )
        envelope = genesis_payload.get("attestation")
        empty_state = _superseded_empty_quality_state()
        _require(
            genesis_opened.sha256 == V1_3_SUPERSEDED_GENESIS_SHA256
            and genesis_opened.bytes == V1_3_SUPERSEDED_GENESIS_BYTES
            and genesis_payload.get("payload_sha256") == V1_3_SUPERSEDED_GENESIS_PAYLOAD_SHA256
            and isinstance(envelope, Mapping)
            and dict(cast(Mapping[str, Any], envelope))
            == {
                "scheme": "hmac-sha256-v1",
                "purpose": PREHELDOUT_GENESIS_PURPOSE,
                "key_id": trust_root.key_id,
                "payload_sha256": V1_3_SUPERSEDED_GENESIS_ATTESTATION_PAYLOAD_SHA256,
                "mac": V1_3_SUPERSEDED_GENESIS_ATTESTATION_MAC,
            }
            and genesis_payload.get("experiment_id") == QUALITY_EXPERIMENT_ID
            and genesis_payload.get("canonical_path") == str(genesis_path)
            and genesis_payload.get("quality_source")
            == {"commit": V1_3_SUPERSEDED_RESULT_SOURCE_COMMIT, "dirty": False}
            and genesis_payload.get("quality_manifest") == manifest_binding
            and genesis_payload.get("reuse_admission") == admission_binding
            and genesis_payload.get("expected_shards") == EXPECTED_QUALITY_SHARDS
            and genesis_payload.get("coordinate_digest") == QUALITY_COORDINATE_DIGEST
            and tuple(genesis_payload.get("exact_fill_arm_names", ()))
            == FROZEN_EXACT_FILL_ARM_NAMES
            and all(genesis_payload.get(field) == value for field, value in empty_state.items()),
            "Superseded v1.3 genesis is not the exact signed empty prefix.",
        )
        genesis_opened.assert_unchanged()
    finally:
        genesis_opened.close()
    old_absent = _assert_prospective_quality_paths_absent(repository_root=root)
    _assert_canonical_nonobservation_paths_absent(repository_root=root)
    source = {
        "schema_version": 1,
        "lineage_type": "signed-superseded-empty-quality-prefix",
        "experiment_id": QUALITY_EXPERIMENT_ID,
        "result_source_commit": V1_3_SUPERSEDED_RESULT_SOURCE_COMMIT,
        "result_source_tree": V1_3_SUPERSEDED_RESULT_SOURCE_TREE,
        "implementation_source_tree": V1_3_SUPERSEDED_IMPLEMENTATION_SOURCE_TREE,
        "manifest": manifest_binding,
        "reuse_admission": admission_binding,
        "preheldout_genesis": genesis_binding,
        "attestation_key_id": trust_root.key_id,
        "historical_receipt_payload_sha256": (V1_3_SUPERSEDED_HISTORICAL_RECEIPT_PAYLOAD_SHA256),
        "canonical_nonobservation_payload_sha256": (
            V1_3_SUPERSEDED_CANONICAL_NONOBSERVATION_PAYLOAD_SHA256
        ),
        "superseded_prospective_absent_paths": list(old_absent),
        "quality_state": _superseded_empty_quality_state(),
    }
    public_binding = {**source, "lineage_sha256": _json_digest(source)}
    _validate_admission_bundle_root(admission_root, label="Superseded v1.3 admission root")
    return SupersededEmptyLineageV1_3(
        _seal=_SUPERSEDED_EMPTY_LINEAGE_SEAL,
        manifest=manifest_payload,
        reuse_admission=admission_payload,
        preheldout_genesis=genesis_payload,
        public_binding=public_binding,
    )


def _v1_3_1_opened_file_identity(
    opened: attestation.OpenedRegularFile,
) -> dict[str, int]:
    metadata = os.fstat(opened.file_descriptor)
    return {
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mode": metadata.st_mode,
        "uid": metadata.st_uid,
        "gid": metadata.st_gid,
        "nlink": metadata.st_nlink,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
    }


def _v1_3_1_exact_regular_binding(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> dict[str, Any]:
    opened = _open_secure_regular(path, label=label)
    try:
        _require(
            opened.sha256 == expected_sha256 and opened.bytes == expected_bytes,
            f"{label} is not the byte-exact v1.3.1 failure-lineage artifact.",
        )
        opened.assert_unchanged()
        return {
            "path": str(path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
            **_v1_3_1_opened_file_identity(opened),
        }
    finally:
        opened.close()


def _v1_3_1_exact_attested_json(
    path: Path,
    *,
    trust_root: attestation.TrustRoot,
    expected_sha256: str,
    expected_bytes: int,
    expected_payload_sha256: str,
    expected_attestation_payload_sha256: str,
    expected_attestation_mac: str,
    purpose: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, opened = _load_json_nofollow(
        path,
        label=label,
        require_canonical_pretty_bytes=True,
    )
    try:
        _verify_attested_payload(
            payload,
            trust_root=trust_root,
            purpose=purpose,
            label=label,
        )
        envelope = payload.get("attestation")
        expected_envelope = {
            "scheme": "hmac-sha256-v1",
            "purpose": purpose,
            "key_id": trust_root.key_id,
            "payload_sha256": expected_attestation_payload_sha256,
            "mac": expected_attestation_mac,
        }
        _require(
            opened.sha256 == expected_sha256
            and opened.bytes == expected_bytes
            and payload.get("payload_sha256") == expected_payload_sha256
            and isinstance(envelope, Mapping)
            and dict(cast(Mapping[str, Any], envelope)) == expected_envelope,
            f"{label} byte, payload, or HMAC binding drifted.",
        )
        opened.assert_unchanged()
        return dict(payload), {
            "path": str(path),
            "sha256": opened.sha256,
            "bytes": opened.bytes,
            "payload_sha256": expected_payload_sha256,
            "attestation_payload_sha256": expected_attestation_payload_sha256,
            "attestation_mac": expected_attestation_mac,
            "attestation_purpose": purpose,
            **_v1_3_1_opened_file_identity(opened),
        }
    finally:
        opened.close()


def _v1_3_1_revalidate_regular_binding(
    binding: Mapping[str, Any], *, label: str
) -> None:
    path = binding.get("path")
    _require(
        isinstance(path, str)
        and _is_sha256(binding.get("sha256"))
        and type(binding.get("bytes")) is int,
        f"{label} initial binding is malformed.",
    )
    observed = _v1_3_1_exact_regular_binding(
        Path(cast(str, path)),
        expected_sha256=cast(str, binding["sha256"]),
        expected_bytes=cast(int, binding["bytes"]),
        label=f"{label} final reread",
    )
    _require(
        all(binding.get(field) == value for field, value in observed.items()),
        f"{label} metadata or open-file identity changed before capability return.",
    )


def _v1_3_1_revalidate_attested_binding(
    binding: Mapping[str, Any],
    expected_payload: Mapping[str, Any],
    *,
    trust_root: attestation.TrustRoot,
    label: str,
) -> None:
    required = (
        "path",
        "sha256",
        "bytes",
        "payload_sha256",
        "attestation_payload_sha256",
        "attestation_mac",
        "attestation_purpose",
    )
    _require(
        all(field in binding for field in required),
        f"{label} initial attested binding is incomplete.",
    )
    payload, observed = _v1_3_1_exact_attested_json(
        Path(cast(str, binding["path"])),
        trust_root=trust_root,
        expected_sha256=cast(str, binding["sha256"]),
        expected_bytes=cast(int, binding["bytes"]),
        expected_payload_sha256=cast(str, binding["payload_sha256"]),
        expected_attestation_payload_sha256=cast(
            str, binding["attestation_payload_sha256"]
        ),
        expected_attestation_mac=cast(str, binding["attestation_mac"]),
        purpose=cast(str, binding["attestation_purpose"]),
        label=f"{label} final reread",
    )
    _require(
        payload == dict(expected_payload) and observed == dict(binding),
        f"{label} payload, HMAC, metadata, or identity changed before capability return.",
    )


def _v1_3_1_activation_public_binding(
    *, path: Path, payload: Mapping[str, Any], sha256: str, byte_count: int
) -> dict[str, Any]:
    """Reconstruct the historical v1.3.1 activation reference exactly."""

    envelope = payload.get("attestation")
    lock = payload.get("matrix_lock_binding")
    source = payload.get("sealed_source_provenance")
    routing = payload.get("sealed_launch_routing")
    _require(
        all(isinstance(value, Mapping) for value in (envelope, lock, source, routing)),
        "Superseded v1.3.1 activation public-binding inputs are invalid.",
    )
    return {
        "path": str(path),
        "sha256": sha256,
        "bytes": byte_count,
        "experiment_id": payload.get("experiment_id"),
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": cast(Mapping[str, Any], envelope).get("mac"),
        "activation_root": payload.get("canonical_root"),
        "matrix_lock_path": cast(Mapping[str, Any], lock).get("path"),
        "matrix_lock_device": cast(Mapping[str, Any], lock).get("device"),
        "matrix_lock_inode": cast(Mapping[str, Any], lock).get("inode"),
        "base_prerequisites_sha256": payload.get("base_prerequisites_sha256"),
        "sealed_source_bundle_sha256": cast(Mapping[str, Any], source).get(
            "bundle_sha256"
        ),
        "sealed_launch_routing_sha256": _json_digest(
            dict(cast(Mapping[str, Any], routing))
        ),
    }


def _v1_3_1_reuse_admission_public_binding(
    *, path: Path, payload: Mapping[str, Any], sha256: str, byte_count: int
) -> dict[str, Any]:
    """Reconstruct the historical v1.3.1 admission reference exactly."""

    envelope = payload.get("attestation")
    lineage = payload.get("superseded_empty_lineage")
    _require(
        isinstance(envelope, Mapping) and isinstance(lineage, Mapping),
        "Superseded v1.3.1 reuse-admission public-binding inputs are invalid.",
    )
    lineage_map = cast(Mapping[str, Any], lineage)
    return {
        "path": str(path),
        "sha256": sha256,
        "bytes": byte_count,
        "experiment_id": payload.get("experiment_id"),
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": cast(Mapping[str, Any], envelope).get("mac"),
        "historical_receipt_sha256": lineage_map.get(
            "historical_receipt_payload_sha256"
        ),
        "canonical_nonobservation_sha256": lineage_map.get(
            "canonical_nonobservation_payload_sha256"
        ),
    }


def _v1_3_1_closed_world_inventory(
    root: Path,
    *,
    expected_directories: set[str],
    expected_files: set[str],
    label: str,
) -> dict[str, Any]:
    root = _exact_path(root, label=label, must_exist=True)
    directories: set[str] = set()
    files: set[str] = set()
    directory_snapshots: dict[str, dict[str, Any]] = {}

    def snapshot(metadata: os.stat_result, entries: tuple[str, ...]) -> dict[str, Any]:
        return {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "mode": metadata.st_mode,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "nlink": metadata.st_nlink,
            "size": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
            "entries": list(entries),
        }

    def scan(directory: Path, relative: str) -> None:
        before = os.stat(directory, follow_symlinks=False)
        _require(
            stat.S_ISDIR(before.st_mode)
            and before.st_uid == os.getuid()
            and stat.S_IMODE(before.st_mode) == SAFE_DIRECTORY_MODE,
            f"{label} directory metadata is unsafe: {directory}",
        )
        directories.add(relative)
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        entry_names = tuple(entry.name for entry in entries)
        for entry in entries:
            path = Path(entry.path)
            metadata = entry.stat(follow_symlinks=False)
            child_relative = entry.name if relative == "." else f"{relative}/{entry.name}"
            _require(not stat.S_ISLNK(metadata.st_mode), f"{label} contains a symlink: {path}")
            if stat.S_ISDIR(metadata.st_mode):
                scan(_exact_path(path, label=f"{label} directory", must_exist=True), child_relative)
            else:
                _require(
                    stat.S_ISREG(metadata.st_mode)
                    and metadata.st_uid == os.getuid()
                    and metadata.st_nlink == 1
                    and stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE,
                    f"{label} contains an unsafe non-regular file: {path}",
                )
                files.add(child_relative)
        after = os.stat(directory, follow_symlinks=False)
        final_entry_names = tuple(sorted(os.listdir(directory)))
        _require(
            snapshot(before, entry_names) == snapshot(after, final_entry_names),
            f"{label} directory changed during closed-world validation: {directory}",
        )
        directory_snapshots[relative] = snapshot(after, final_entry_names)

    scan(root, ".")
    _require(
        directories == expected_directories and files == expected_files,
        f"{label} closed world drifted: directories={sorted(directories)}, files={sorted(files)}",
    )
    return {
        "root": str(root),
        "exact_relative_directories": sorted(directories),
        "exact_relative_files": sorted(files),
        "directory_snapshots": {
            relative: directory_snapshots[relative]
            for relative in sorted(directory_snapshots)
        },
    }


def _v1_3_1_revalidate_closed_world_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_directories: set[str],
    expected_files: set[str],
    label: str,
) -> None:
    root = inventory.get("root")
    _require(isinstance(root, str), f"{label} inventory root is missing.")
    observed = _v1_3_1_closed_world_inventory(
        Path(cast(str, root)),
        expected_directories=expected_directories,
        expected_files=expected_files,
        label=f"{label} final rescan",
    )
    _require(
        observed == dict(inventory),
        f"{label} changed between its authenticated scan and final rescan.",
    )


def _v1_3_1_claim_owner_is_live(*, pid: int, process_start_ticks: int) -> bool:
    path = Path("/proc") / str(pid) / "stat"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError("Cannot fail-closed while checking the v1.3.1 claim owner.") from error
    _prefix, separator, tail = raw.rpartition(") ")
    fields = tail.split()
    _require(separator == ") " and len(fields) >= 20, "Claim-owner /proc stat is malformed.")
    try:
        observed_start_ticks = int(fields[19])
    except ValueError as error:
        raise ValueError("Claim-owner /proc start time is malformed.") from error
    return observed_start_ticks == process_start_ticks


def _v1_3_1_require_absent_paths_and_dead_owner(
    *,
    absent_paths: Sequence[Path],
    pid: int,
    process_start_ticks: int,
    boundary: str,
) -> None:
    for path in absent_paths:
        _require(
            not os.path.lexists(path),
            f"Forbidden v1.3.1 post-quality artifact exists {boundary}: {path}",
        )
    _require(
        not _v1_3_1_claim_owner_is_live(
            pid=pid,
            process_start_ticks=process_start_ticks,
        ),
        f"Superseded v1.3.1 claim owner is live {boundary}.",
    )


def _v1_3_1_zero_quality_state() -> dict[str, Any]:
    return {
        "records": [],
        "completed_shards": 0,
        "globally_committed_shards": 0,
        "integrity_pass_shards": 0,
        "integrity_fail_shards": 0,
        "quality_evaluation_started": False,
        "evaluation_seed_used_to_initialize_quality_rng": False,
        "quality_rng_initialized": False,
        "evaluation_inputs_materialized": 0,
        "quality_predictions_materialized": 0,
        "quality_outcomes_materialized": 0,
        "quality_aggregates_materialized": 0,
        "quality_outcomes_aggregated": False,
        "outcome_selection_performed": False,
        "outcome_dependent_early_stopping": False,
        "live_active_claim_count": 0,
        "orphan_claim_count": 1,
        "worker_ledger_count": 0,
        "ready_receipt_count": 0,
        "work_order_count": 0,
        "work_result_count": 0,
        "final_receipt_count": 0,
    }


def _v1_3_1_normalized_failure_projection(
    source: Mapping[str, Any], *, repository_root: Path
) -> dict[str, Any]:
    """Normalize the authenticated live lineage to the manifest's relative projection."""

    root = Path(os.path.abspath(repository_root))

    def relative_path(value: object, *, label: str) -> str:
        _require(isinstance(value, str), f"{label} path is missing.")
        path = Path(cast(str, value))
        _require(path.is_absolute(), f"{label} path is not absolute.")
        try:
            return str(path.relative_to(root))
        except ValueError as error:
            raise ValueError(f"{label} path escapes the repository root.") from error

    def projected_artifact(
        value: object, *, fields: tuple[str, ...], label: str
    ) -> dict[str, Any]:
        _require(isinstance(value, Mapping), f"{label} binding is missing.")
        binding = cast(Mapping[str, Any], value)
        _require(all(field in binding for field in fields), f"{label} binding is incomplete.")
        projected = {field: binding[field] for field in fields}
        projected["path"] = relative_path(projected["path"], label=label)
        return projected

    implementation = source.get("implementation")
    activation_root = source.get("activation_root")
    output = source.get("output_closed_world")
    persistent = source.get("persistent_session")
    claim = cast(Mapping[str, Any], output).get("orphan_claim") if isinstance(output, Mapping) else None
    _require(
        all(isinstance(value, Mapping) for value in (implementation, activation_root, output, persistent, claim)),
        "Superseded v1.3.1 normalized projection inputs are incomplete.",
    )
    activation_map = cast(Mapping[str, Any], activation_root)
    output_map = cast(Mapping[str, Any], output)
    persistent_map = cast(Mapping[str, Any], persistent)
    claim_map = cast(Mapping[str, Any], claim)
    attested_fields = (
        "path",
        "sha256",
        "bytes",
        "payload_sha256",
        "attestation_payload_sha256",
        "attestation_mac",
        "attestation_purpose",
    )
    projection_source = {
        "schema_version": source.get("schema_version"),
        "lineage_type": source.get("lineage_type"),
        "experiment_id": source.get("experiment_id"),
        "result_source_commit": source.get("result_source_commit"),
        "result_source_tree": source.get("result_source_tree"),
        "implementation": dict(cast(Mapping[str, Any], implementation)),
        "manifest": projected_artifact(
            source.get("manifest"),
            fields=("path", "sha256", "bytes"),
            label="Superseded v1.3.1 manifest",
        ),
        "reuse_admission": projected_artifact(
            source.get("reuse_admission"),
            fields=attested_fields,
            label="Superseded v1.3.1 reuse admission",
        ),
        "preheldout_genesis": projected_artifact(
            source.get("preheldout_genesis"),
            fields=attested_fields,
            label="Superseded v1.3.1 pre-heldout genesis",
        ),
        "activation_root": {
            "path": relative_path(
                activation_map.get("root"), label="Superseded v1.3.1 activation root"
            ),
            "exact_members": [
                projected_artifact(
                    activation_map.get("matrix_lock"),
                    fields=("path", "sha256", "bytes"),
                    label="Superseded v1.3.1 activation lock",
                ),
                projected_artifact(
                    activation_map.get("quality_start_activation"),
                    fields=attested_fields,
                    label="Superseded v1.3.1 activation",
                ),
            ],
        },
        "output_closed_world": {
            "root": relative_path(
                output_map.get("root"), label="Superseded v1.3.1 output root"
            ),
            "exact_relative_directories": output_map.get("exact_relative_directories"),
            "exact_relative_files": output_map.get("exact_relative_files"),
            "matrix": projected_artifact(
                output_map.get("matrix"),
                fields=attested_fields,
                label="Superseded v1.3.1 matrix",
            ),
            "orphan_claim": {
                "path": relative_path(
                    claim_map.get("path"), label="Superseded v1.3.1 orphan claim"
                ),
                "relative_path": claim_map.get("relative_path"),
                "sha256": claim_map.get("sha256"),
                "bytes": claim_map.get("bytes"),
                "pid": claim_map.get("pid"),
                "process_start_ticks": claim_map.get("process_start_ticks"),
                "dead_owner_required": claim_map.get("dead_owner"),
                "quality_payload_fields_present": claim_map.get(
                    "quality_payload_fields_present"
                ),
            },
        },
        "persistent_session": {
            "root": relative_path(
                persistent_map.get("root"), label="Superseded v1.3.1 session root"
            ),
            "exact_members": [
                projected_artifact(
                    persistent_map.get("launch"),
                    fields=attested_fields,
                    label="Superseded v1.3.1 session launch",
                ),
                projected_artifact(
                    persistent_map.get("terminal"),
                    fields=attested_fields,
                    label="Superseded v1.3.1 session terminal",
                ),
            ],
            "lock": projected_artifact(
                persistent_map.get("lock"),
                fields=("path", "sha256", "bytes"),
                label="Superseded v1.3.1 session lock",
            ),
            "session_nonce": persistent_map.get("session_nonce"),
            "launch_authority_nonce": persistent_map.get("launch_authority_nonce"),
            "terminal_status": persistent_map.get("terminal_status"),
            "child_process_returncode": persistent_map.get("child_process_returncode"),
        },
        "absent_paths": [
            relative_path(path, label="Superseded v1.3.1 absent artifact")
            for path in cast(Sequence[object], source.get("absent_paths"))
        ],
        "attestation_key_id": source.get("attestation_key_id"),
        "quality_state": source.get("quality_state"),
    }
    return {
        **projection_source,
        "projection_sha256": _json_digest(projection_source),
    }


def _require_v1_3_1_normalized_failure_projection(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed on static/live projection drift without needing the HMAC secret."""

    checked = dict(value)
    unsigned = {key: item for key, item in checked.items() if key != "projection_sha256"}
    contract_module = importlib.import_module("p2_direct_controller_contract_v1_3")
    builder = getattr(contract_module, "expected_v1_3_2_superseded_failure_lineage", None)
    _require(callable(builder), "The v1.3.2 failure-lineage contract builder is unavailable.")
    expected = cast(Callable[[], object], builder)()
    _require(
        _is_sha256(checked.get("projection_sha256"))
        and checked.get("projection_sha256") == _json_digest(unsigned)
        and isinstance(expected, Mapping)
        and checked == dict(cast(Mapping[str, Any], expected)),
        "Normalized v1.3.1 failure-lineage projection drifted from the manifest contract.",
    )
    return checked


def _validate_v1_3_1_failure_relations(
    *,
    manifest_public_binding: Mapping[str, Any],
    admission_public_binding: Mapping[str, Any],
    genesis_public_binding: Mapping[str, Any],
    activation_public_binding: Mapping[str, Any],
    inherited_public_binding: Mapping[str, Any],
    activation: Mapping[str, Any],
    matrix: Mapping[str, Any],
    launch: Mapping[str, Any],
    terminal: Mapping[str, Any],
    launch_binding: Mapping[str, Any],
    terminal_binding: Mapping[str, Any],
    session_root: Path,
    session_lock_path: Path,
    session_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate cross-artifact links separately from HMAC and filesystem I/O."""

    prerequisites = matrix.get("prerequisites")
    plan = launch.get("plan")
    _require(
        isinstance(prerequisites, Mapping) and isinstance(plan, Mapping),
        "Superseded v1.3.1 relational prerequisites or session plan are missing.",
    )
    _require(
        activation.get("quality_manifest") == manifest_public_binding
        and activation.get("reuse_admission") == admission_public_binding
        and activation.get("preheldout_genesis") == genesis_public_binding
        and activation.get("superseded_empty_lineage") == inherited_public_binding
        and matrix.get("manifest") == manifest_public_binding
        and matrix.get("source")
        == {"commit": V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT, "dirty": False}
        and cast(Mapping[str, Any], prerequisites).get("manifest")
        == manifest_public_binding
        and cast(Mapping[str, Any], prerequisites).get("reuse_admission")
        == admission_public_binding
        and cast(Mapping[str, Any], prerequisites).get("preheldout_genesis")
        == genesis_public_binding
        and cast(Mapping[str, Any], prerequisites).get("quality_start_activation")
        == activation_public_binding
        and matrix.get("matrix_lock") == activation.get("matrix_lock_binding"),
        "Superseded v1.3.1 cross-artifact public or matrix-lock relation drifted.",
    )
    _require(
        launch.get("actual_session_argv") == terminal.get("actual_session_argv")
        and terminal.get("plan_payload_sha256")
        == cast(Mapping[str, Any], plan).get("payload_sha256"),
        "Superseded v1.3.1 launch and terminal relation drifted.",
    )
    session_nonce = launch.get("session_nonce")
    expected_registry = [
        {
            "kind": kind,
            "path": binding["path"],
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "payload_sha256": binding["payload_sha256"],
            "attestation_mac": binding["attestation_mac"],
            "session_nonce": session_nonce,
        }
        for kind, binding in (("launch", launch_binding), ("terminal", terminal_binding))
    ]
    expected_session = {
        "actual_session_argv": launch["actual_session_argv"],
        "child_process_returncode": terminal["child_process_returncode"],
        "completed_result_payload_sha256": terminal["completed_result_payload_sha256"],
        "completed_work_payload_sha256": terminal["completed_work_payload_sha256"],
        "coordinate_count": cast(Mapping[str, Any], plan)["coordinate_count"],
        "coordinate_digest": cast(Mapping[str, Any], plan)["coordinate_digest"],
        "launch_authority_nonce": launch["launch_authority_nonce"],
        "max_new_cells_stop_limit": cast(Mapping[str, Any], plan)[
            "max_new_cells_stop_limit"
        ],
        "plan_payload_sha256": cast(Mapping[str, Any], plan)["payload_sha256"],
        "published_bundle_reingestion_count": terminal[
            "published_bundle_reingestion_count"
        ],
        "ready_model_load_observed": False,
        "scale": cast(Mapping[str, Any], plan)["scale"],
        "session_nonce": session_nonce,
        "status": terminal["status"],
        "training_seed": cast(Mapping[str, Any], plan)["training_seed"],
        "worker_count": cast(Mapping[str, Any], plan)["worker_count"],
        "worker_index": cast(Mapping[str, Any], plan)["worker_index"],
    }
    expected_counters = {
        "launch_attempt_count": 1,
        "ready_model_load_count": 0,
        "observed_successful_model_loads": 0,
        "terminal_count": 1,
        "published_bundle_reingestion_count": 0,
        "launch_authority_count": 1,
        "controlled_stop_session_count": 1,
    }
    ledger = matrix.get("persistent_session_ledger")
    _require(
        isinstance(ledger, Mapping),
        "Superseded v1.3.1 persistent-session relational ledger is missing.",
    )
    ledger_map = cast(Mapping[str, Any], ledger)
    matrix_session_root = ledger_map.get("root")
    _require(
        matrix_session_root == str(session_root)
        and ledger_map.get("registry") == expected_registry
        and ledger_map.get("registry_digest") == _json_digest(expected_registry)
        and ledger_map.get("sessions") == [expected_session]
        and ledger_map.get("sessions_digest") == _json_digest([expected_session])
        and all(ledger_map.get(field) == value for field, value in expected_counters.items())
        and session_lock_path == Path(f"{matrix_session_root}.lock")
        and session_lock_binding.get("path") == str(session_lock_path),
        "Superseded v1.3.1 matrix session relation drifted.",
    )
    return {
        "registry": expected_registry,
        "session": expected_session,
        "counters": expected_counters,
        "registry_digest": ledger_map["registry_digest"],
        "sessions_digest": ledger_map["sessions_digest"],
    }


def load_superseded_zero_quality_failure_lineage_v1_3_1(
    *,
    trust_root: attestation.TrustRoot,
    repository_root: Path = REPOSITORY_ROOT,
    superseded_empty_lineage: SupersededEmptyLineageV1_3 | None = None,
) -> SupersededZeroQualityFailureLineageV1_3_1:
    """Authenticate the immutable v1.3.1 launch failure and prove zero quality."""

    _require(
        type(trust_root) is attestation.TrustRoot
        and trust_root.key_id == V1_3_SUPERSEDED_ATTESTATION_KEY_ID,
        "Superseded v1.3.1 lineage requires its exact historical trust root.",
    )
    root = _exact_path(repository_root, label="Repository root", must_exist=True)
    inherited = (
        load_superseded_empty_lineage_v1_3(trust_root=trust_root, repository_root=root)
        if superseded_empty_lineage is None
        else _require_superseded_lineage(superseded_empty_lineage)
    )
    _require(
        inherited.public_binding.get("lineage_sha256") == V1_3_SUPERSEDED_LINEAGE_SHA256,
        "The v1.3.1 failure does not descend from the exact v1.3 signed-empty lineage.",
    )
    current_commit = cast(str, _source_state(root)["commit"])
    _assert_ancestor(root, V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT, current_commit)
    _assert_tree_object(
        root,
        V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT,
        V1_3_1_SUPERSEDED_RESULT_SOURCE_TREE,
    )
    _assert_tree_object(
        root,
        V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT,
        V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_TREE,
    )

    manifest_path = _absolute(V1_3_1_SUPERSEDED_MANIFEST_RELATIVE_PATH, repository_root=root)
    manifest, manifest_opened = _load_json_nofollow(
        manifest_path,
        label="Superseded v1.3.1 manifest",
        require_canonical_pretty_bytes=True,
    )
    try:
        implementation = manifest.get("implementation")
        manifest_attestation = manifest.get("attestation")
        _require(
            manifest_opened.sha256 == V1_3_1_SUPERSEDED_MANIFEST_SHA256
            and manifest_opened.bytes == V1_3_1_SUPERSEDED_MANIFEST_BYTES
            and manifest.get("schema_version") == 1
            and manifest.get("experiment_id") == V1_3_1_SUPERSEDED_EXPERIMENT_ID
            and manifest.get("status")
            == (
                "frozen_v1_3_1_activation_amendment_after_signed_v1_3_empty_prefix_"
                "before_any_held_out_quality"
            )
            and isinstance(implementation, Mapping)
            and cast(Mapping[str, Any], implementation).get("source_commit")
            == V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT
            and cast(Mapping[str, Any], implementation).get("tree_digest")
            == V1_3_1_SUPERSEDED_IMPLEMENTATION_DIGEST
            and isinstance(manifest_attestation, Mapping)
            and cast(Mapping[str, Any], manifest_attestation).get("key_id") == trust_root.key_id,
            "Superseded v1.3.1 manifest binding drifted.",
        )
        manifest_binding = {
            "path": str(manifest_path),
            "sha256": manifest_opened.sha256,
            "bytes": manifest_opened.bytes,
            **_v1_3_1_opened_file_identity(manifest_opened),
            "experiment_id": V1_3_1_SUPERSEDED_EXPERIMENT_ID,
            "implementation_source_commit": V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT,
            "implementation_source_tree": V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_TREE,
            "implementation_digest": V1_3_1_SUPERSEDED_IMPLEMENTATION_DIGEST,
        }
        manifest_opened.assert_unchanged()
    finally:
        manifest_opened.close()

    admission_root = _absolute(V1_3_1_SUPERSEDED_ADMISSION_ROOT, repository_root=root)
    admission_path = _absolute(V1_3_1_SUPERSEDED_ADMISSION_PATH, repository_root=root)
    genesis_path = _absolute(V1_3_1_SUPERSEDED_GENESIS_PATH, repository_root=root)
    admission_expected_directories = {"."}
    admission_expected_files = {admission_path.name, genesis_path.name}
    admission_inventory = _v1_3_1_closed_world_inventory(
        admission_root,
        expected_directories=admission_expected_directories,
        expected_files=admission_expected_files,
        label="Superseded v1.3.1 admission root",
    )
    admission, admission_binding = _v1_3_1_exact_attested_json(
        admission_path,
        trust_root=trust_root,
        expected_sha256=V1_3_1_SUPERSEDED_ADMISSION_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_ADMISSION_BYTES,
        expected_payload_sha256=V1_3_1_SUPERSEDED_ADMISSION_PAYLOAD_SHA256,
        expected_attestation_payload_sha256=(
            V1_3_1_SUPERSEDED_ADMISSION_ATTESTATION_PAYLOAD_SHA256
        ),
        expected_attestation_mac=V1_3_1_SUPERSEDED_ADMISSION_ATTESTATION_MAC,
        purpose=V1_3_1_SUPERSEDED_REUSE_ADMISSION_PURPOSE,
        label="Superseded v1.3.1 reuse admission",
    )
    genesis, genesis_binding = _v1_3_1_exact_attested_json(
        genesis_path,
        trust_root=trust_root,
        expected_sha256=V1_3_1_SUPERSEDED_GENESIS_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_GENESIS_BYTES,
        expected_payload_sha256=V1_3_1_SUPERSEDED_GENESIS_PAYLOAD_SHA256,
        expected_attestation_payload_sha256=(
            V1_3_1_SUPERSEDED_GENESIS_ATTESTATION_PAYLOAD_SHA256
        ),
        expected_attestation_mac=V1_3_1_SUPERSEDED_GENESIS_ATTESTATION_MAC,
        purpose=V1_3_1_SUPERSEDED_PREHELDOUT_GENESIS_PURPOSE,
        label="Superseded v1.3.1 pre-heldout genesis",
    )
    manifest_public_binding_value = admission.get("quality_manifest")
    _require(
        isinstance(manifest_public_binding_value, Mapping),
        "Superseded v1.3.1 manifest public binding is missing.",
    )
    manifest_public_binding = dict(cast(Mapping[str, Any], manifest_public_binding_value))
    manifest_public_attestation = manifest_public_binding.get("attestation")
    admission_public_binding = _v1_3_1_reuse_admission_public_binding(
        path=admission_path,
        payload=admission,
        sha256=V1_3_1_SUPERSEDED_ADMISSION_SHA256,
        byte_count=V1_3_1_SUPERSEDED_ADMISSION_BYTES,
    )
    genesis_public_binding = _genesis_public_binding(
        path=genesis_path,
        payload=genesis,
        sha256=V1_3_1_SUPERSEDED_GENESIS_SHA256,
        byte_count=V1_3_1_SUPERSEDED_GENESIS_BYTES,
    )
    _require(
        manifest_public_binding.get("path") == str(manifest_path)
        and manifest_public_binding.get("sha256") == V1_3_1_SUPERSEDED_MANIFEST_SHA256
        and manifest_public_binding.get("bytes") == V1_3_1_SUPERSEDED_MANIFEST_BYTES
        and manifest_public_binding.get("experiment_id") == V1_3_1_SUPERSEDED_EXPERIMENT_ID
        and manifest_public_binding.get("implementation_source_commit")
        == V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT
        and manifest_public_binding.get("implementation_digest")
        == V1_3_1_SUPERSEDED_IMPLEMENTATION_DIGEST
        and isinstance(manifest_public_attestation, Mapping)
        and cast(Mapping[str, Any], manifest_public_attestation).get("key_id")
        == trust_root.key_id
        and genesis.get("quality_manifest") == manifest_public_binding
        and genesis.get("reuse_admission") == admission_public_binding,
        "Superseded v1.3.1 manifest/admission/genesis public bindings are not relationally exact.",
    )
    zero_static_fields = (
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "active_claim_count",
        "worker_ledger_count",
        "top_p_quality_input_count",
    )
    _require(
        admission.get("experiment_id") == V1_3_1_SUPERSEDED_EXPERIMENT_ID
        and admission.get("canonical_path") == str(admission_path)
        and admission.get("quality_source")
        == {"commit": V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT, "dirty": False}
        and isinstance(admission.get("quality_manifest"), Mapping)
        and cast(Mapping[str, Any], admission["quality_manifest"]).get("sha256")
        == V1_3_1_SUPERSEDED_MANIFEST_SHA256
        and admission.get("superseded_empty_lineage") == inherited.public_binding
        and admission.get("quality_evaluation_started") is False
        and admission.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and admission.get("quality_rng_initialized") is False
        and all(admission.get(field) == 0 for field in zero_static_fields)
        and admission.get("scientific_subprocesses_started_during_admission") == 0,
        "Superseded v1.3.1 admission is not an exact zero-quality static artifact.",
    )
    _require(
        genesis.get("experiment_id") == V1_3_1_SUPERSEDED_EXPERIMENT_ID
        and genesis.get("canonical_path") == str(genesis_path)
        and genesis.get("quality_source")
        == {"commit": V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT, "dirty": False}
        and isinstance(genesis.get("quality_manifest"), Mapping)
        and cast(Mapping[str, Any], genesis["quality_manifest"]).get("sha256")
        == V1_3_1_SUPERSEDED_MANIFEST_SHA256
        and isinstance(genesis.get("reuse_admission"), Mapping)
        and cast(Mapping[str, Any], genesis["reuse_admission"]).get("sha256")
        == V1_3_1_SUPERSEDED_ADMISSION_SHA256
        and genesis.get("superseded_empty_lineage") == inherited.public_binding
        and genesis.get("records") == []
        and genesis.get("completed_shards") == 0
        and genesis.get("quality_evaluation_started") is False
        and genesis.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and genesis.get("quality_rng_initialized") is False
        and all(genesis.get(field) == 0 for field in zero_static_fields),
        "Superseded v1.3.1 genesis is not the exact zero-quality prefix.",
    )

    activation_root = _absolute(V1_3_1_SUPERSEDED_ACTIVATION_ROOT, repository_root=root)
    activation_lock_path = _absolute(
        V1_3_1_SUPERSEDED_ACTIVATION_LOCK_PATH,
        repository_root=root,
    )
    activation_path = _absolute(V1_3_1_SUPERSEDED_ACTIVATION_PATH, repository_root=root)
    activation_expected_directories = {"."}
    activation_expected_files = {activation_lock_path.name, activation_path.name}
    activation_inventory = _v1_3_1_closed_world_inventory(
        activation_root,
        expected_directories=activation_expected_directories,
        expected_files=activation_expected_files,
        label="Superseded v1.3.1 activation root",
    )
    activation_lock_binding = _v1_3_1_exact_regular_binding(
        activation_lock_path,
        expected_sha256=V1_3_1_SUPERSEDED_ACTIVATION_LOCK_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_ACTIVATION_LOCK_BYTES,
        label="Superseded v1.3.1 activation matrix lock",
    )
    activation, activation_binding = _v1_3_1_exact_attested_json(
        activation_path,
        trust_root=trust_root,
        expected_sha256=V1_3_1_SUPERSEDED_ACTIVATION_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_ACTIVATION_BYTES,
        expected_payload_sha256=V1_3_1_SUPERSEDED_ACTIVATION_PAYLOAD_SHA256,
        expected_attestation_payload_sha256=(
            V1_3_1_SUPERSEDED_ACTIVATION_ATTESTATION_PAYLOAD_SHA256
        ),
        expected_attestation_mac=V1_3_1_SUPERSEDED_ACTIVATION_ATTESTATION_MAC,
        purpose=V1_3_1_SUPERSEDED_ACTIVATION_PURPOSE,
        label="Superseded v1.3.1 quality-start activation",
    )
    activation_public_binding = _v1_3_1_activation_public_binding(
        path=activation_path,
        payload=activation,
        sha256=V1_3_1_SUPERSEDED_ACTIVATION_SHA256,
        byte_count=V1_3_1_SUPERSEDED_ACTIVATION_BYTES,
    )
    _require(
        activation.get("experiment_id") == V1_3_1_SUPERSEDED_EXPERIMENT_ID
        and activation.get("status") == "activated"
        and activation.get("canonical_path") == str(activation_path)
        and activation.get("canonical_root") == str(activation_root)
        and activation.get("quality_source")
        == {"commit": V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT, "dirty": False}
        and activation.get("records") == []
        and activation.get("completed_shards") == 0
        and activation.get("quality_evaluation_started") is False
        and activation.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and activation.get("quality_rng_initialized") is False
        and all(activation.get(field) == 0 for field in zero_static_fields)
        and activation.get("scientific_subprocesses_started_during_activation") == 0,
        "Superseded v1.3.1 activation is not the exact signed zero-quality activation.",
    )
    _require(
        activation.get("quality_manifest") == manifest_public_binding
        and activation.get("reuse_admission") == admission_public_binding
        and activation.get("preheldout_genesis") == genesis_public_binding
        and activation.get("superseded_empty_lineage") == inherited.public_binding,
        "Superseded v1.3.1 activation prerequisite public bindings drifted.",
    )

    output_root = _absolute(V1_3_1_SUPERSEDED_OUTPUT_ROOT, repository_root=root)
    matrix_path = _absolute(V1_3_1_SUPERSEDED_MATRIX_PATH, repository_root=root)
    claim_path = _absolute(V1_3_1_SUPERSEDED_CLAIM_PATH, repository_root=root)
    output_expected_directories = {
        ".",
        "s55",
        "s55/seed-6071406",
        "s55/seed-6071406/2x",
        "s55/seed-6071406/2x/single-remote-retrieval",
        "s55/seed-6071406/2x/single-remote-retrieval/context-80",
        "s55/seed-6071406/2x/single-remote-retrieval/context-80/replicate-0",
    }
    output_expected_files = {
        matrix_path.name,
        str(V1_3_1_SUPERSEDED_CLAIM_RELATIVE_PATH),
    }
    output_inventory = _v1_3_1_closed_world_inventory(
        output_root,
        expected_directories=output_expected_directories,
        expected_files=output_expected_files,
        label="Superseded v1.3.1 quality output",
    )
    matrix, matrix_binding = _v1_3_1_exact_attested_json(
        matrix_path,
        trust_root=trust_root,
        expected_sha256=V1_3_1_SUPERSEDED_MATRIX_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_MATRIX_BYTES,
        expected_payload_sha256=V1_3_1_SUPERSEDED_MATRIX_PAYLOAD_SHA256,
        expected_attestation_payload_sha256=(
            V1_3_1_SUPERSEDED_MATRIX_ATTESTATION_PAYLOAD_SHA256
        ),
        expected_attestation_mac=V1_3_1_SUPERSEDED_MATRIX_ATTESTATION_MAC,
        purpose=V1_3_1_SUPERSEDED_MATRIX_PURPOSE,
        label="Superseded v1.3.1 zero-record matrix",
    )
    _require(
        matrix.get("experiment_id") == V1_3_1_SUPERSEDED_MATRIX_EXPERIMENT_ID
        and matrix.get("status") == "in_progress"
        and matrix.get("records") == []
        and matrix.get("completed_shards") == 0
        and matrix.get("globally_committed_shards") == 0
        and matrix.get("integrity_pass_shards") == 0
        and matrix.get("integrity_fail_shards") == 0
        and matrix.get("canonical_prefix_shards") == 0
        and matrix.get("quality_outcomes_aggregated") is False
        and matrix.get("outcome_selection_performed") is False
        and matrix.get("outcome_dependent_early_stopping") is False
        and matrix.get("worker_count") == 1
        and matrix.get("worker_ledger_registry") == [],
        "Superseded v1.3.1 matrix contains quality, selection, or committed work.",
    )
    matrix_prerequisites = matrix.get("prerequisites")
    _require(
        isinstance(matrix_prerequisites, Mapping)
        and matrix.get("manifest") == manifest_public_binding
        and matrix.get("source")
        == {"commit": V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT, "dirty": False}
        and cast(Mapping[str, Any], matrix_prerequisites).get("manifest")
        == manifest_public_binding
        and cast(Mapping[str, Any], matrix_prerequisites).get("reuse_admission")
        == admission_public_binding
        and cast(Mapping[str, Any], matrix_prerequisites).get("preheldout_genesis")
        == genesis_public_binding
        and cast(Mapping[str, Any], matrix_prerequisites).get("quality_start_activation")
        == activation_public_binding
        and matrix.get("matrix_lock") == activation.get("matrix_lock_binding"),
        "Superseded v1.3.1 matrix prerequisite public bindings drifted.",
    )

    claim, claim_opened = _load_json_nofollow(
        claim_path,
        label="Superseded v1.3.1 residual claim",
        require_canonical_pretty_bytes=False,
    )
    try:
        _require(
            claim_opened.sha256 == V1_3_1_SUPERSEDED_CLAIM_SHA256
            and claim_opened.bytes == V1_3_1_SUPERSEDED_CLAIM_BYTES
            and set(claim)
            == {
                "schema_version",
                "semantics",
                "coordinate",
                "created_time_ns",
                "launch_nonce",
                "pid",
                "process_start_ticks",
                "worker_count",
                "worker_index",
            }
            and claim.get("schema_version") == 1
            and claim.get("semantics") == "exclusive-create-coordinate-nonce-claim-v1"
            and claim.get("pid") == V1_3_1_SUPERSEDED_CLAIM_PID
            and claim.get("process_start_ticks")
            == V1_3_1_SUPERSEDED_CLAIM_PROCESS_START_TICKS,
            "Superseded v1.3.1 residual claim drifted or contains non-coordination data.",
        )
        claim_file_identity = _v1_3_1_opened_file_identity(claim_opened)
        claim_opened.assert_unchanged()
    finally:
        claim_opened.close()
    _require(
        not _v1_3_1_claim_owner_is_live(
            pid=V1_3_1_SUPERSEDED_CLAIM_PID,
            process_start_ticks=V1_3_1_SUPERSEDED_CLAIM_PROCESS_START_TICKS,
        ),
        "Superseded v1.3.1 residual claim still has its exact live owner.",
    )
    claim_binding = {
        "path": str(claim_path),
        "relative_path": str(V1_3_1_SUPERSEDED_CLAIM_RELATIVE_PATH),
        "sha256": V1_3_1_SUPERSEDED_CLAIM_SHA256,
        "bytes": V1_3_1_SUPERSEDED_CLAIM_BYTES,
        **claim_file_identity,
        "pid": V1_3_1_SUPERSEDED_CLAIM_PID,
        "process_start_ticks": V1_3_1_SUPERSEDED_CLAIM_PROCESS_START_TICKS,
        "dead_owner": True,
        "quality_payload_fields_present": False,
        "coordinate": claim["coordinate"],
    }

    session_root = _absolute(V1_3_1_SUPERSEDED_SESSION_ROOT, repository_root=root)
    launch_path = _absolute(V1_3_1_SUPERSEDED_SESSION_LAUNCH_PATH, repository_root=root)
    terminal_path = _absolute(V1_3_1_SUPERSEDED_SESSION_TERMINAL_PATH, repository_root=root)
    session_expected_directories = {"."}
    session_expected_files = {launch_path.name, terminal_path.name}
    session_inventory = _v1_3_1_closed_world_inventory(
        session_root,
        expected_directories=session_expected_directories,
        expected_files=session_expected_files,
        label="Superseded v1.3.1 persistent-session root",
    )
    launch, launch_binding = _v1_3_1_exact_attested_json(
        launch_path,
        trust_root=trust_root,
        expected_sha256=V1_3_1_SUPERSEDED_SESSION_LAUNCH_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_SESSION_LAUNCH_BYTES,
        expected_payload_sha256=V1_3_1_SUPERSEDED_SESSION_LAUNCH_PAYLOAD_SHA256,
        expected_attestation_payload_sha256=(
            V1_3_1_SUPERSEDED_SESSION_LAUNCH_ATTESTATION_PAYLOAD_SHA256
        ),
        expected_attestation_mac=V1_3_1_SUPERSEDED_SESSION_LAUNCH_ATTESTATION_MAC,
        purpose=V1_3_1_SUPERSEDED_SESSION_LAUNCH_PURPOSE,
        label="Superseded v1.3.1 persistent launch",
    )
    terminal, terminal_binding = _v1_3_1_exact_attested_json(
        terminal_path,
        trust_root=trust_root,
        expected_sha256=V1_3_1_SUPERSEDED_SESSION_TERMINAL_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_SESSION_TERMINAL_BYTES,
        expected_payload_sha256=V1_3_1_SUPERSEDED_SESSION_TERMINAL_PAYLOAD_SHA256,
        expected_attestation_payload_sha256=(
            V1_3_1_SUPERSEDED_SESSION_TERMINAL_ATTESTATION_PAYLOAD_SHA256
        ),
        expected_attestation_mac=V1_3_1_SUPERSEDED_SESSION_TERMINAL_ATTESTATION_MAC,
        purpose=V1_3_1_SUPERSEDED_SESSION_TERMINAL_PURPOSE,
        label="Superseded v1.3.1 persistent terminal",
    )
    session_lock_path = _absolute(V1_3_1_SUPERSEDED_SESSION_LOCK_PATH, repository_root=root)
    session_lock_binding = _v1_3_1_exact_regular_binding(
        session_lock_path,
        expected_sha256=V1_3_1_SUPERSEDED_SESSION_LOCK_SHA256,
        expected_bytes=V1_3_1_SUPERSEDED_SESSION_LOCK_BYTES,
        label="Superseded v1.3.1 persistent-session lock",
    )
    plan = launch.get("plan")
    _require(isinstance(plan, Mapping), "Superseded v1.3.1 session plan is missing.")
    _verify_attested_payload(
        cast(Mapping[str, Any], plan),
        trust_root=trust_root,
        purpose=V1_3_1_SUPERSEDED_SESSION_PLAN_PURPOSE,
        label="Superseded v1.3.1 persistent-session plan",
    )
    _require(
        launch.get("session_nonce") == V1_3_1_SUPERSEDED_SESSION_NONCE
        and launch.get("launch_authority_nonce") == V1_3_1_SUPERSEDED_LAUNCH_AUTHORITY_NONCE
        and launch.get("outcome_dependent_selection") is False
        and cast(Mapping[str, Any], plan).get("coordinate_count") == 1
        and cast(Mapping[str, Any], plan).get("coordinates") == [claim["coordinate"]]
        and cast(Mapping[str, Any], plan).get("outcome_dependent_selection") is False
        and terminal.get("session_nonce") == V1_3_1_SUPERSEDED_SESSION_NONCE
        and terminal.get("launch_authority_nonce")
        == V1_3_1_SUPERSEDED_LAUNCH_AUTHORITY_NONCE
        and terminal.get("status") == "launch_failure"
        and terminal.get("child_process_returncode") == 1
        and terminal.get("ready_receipt") is None
        and terminal.get("final_receipt") is None
        and terminal.get("completed_work_payload_sha256") == []
        and terminal.get("completed_result_payload_sha256") == []
        and terminal.get("published_bundle_reingestion_count") == 0,
        "Superseded v1.3.1 session is not the exact pre-ready launch failure.",
    )
    _require(
        launch.get("actual_session_argv") == terminal.get("actual_session_argv")
        and terminal.get("plan_payload_sha256")
        == cast(Mapping[str, Any], plan).get("payload_sha256"),
        "Superseded v1.3.1 launch and terminal do not bind the same session plan.",
    )
    expected_session_registry = [
        {
            "kind": kind,
            "path": binding["path"],
            "sha256": binding["sha256"],
            "bytes": binding["bytes"],
            "payload_sha256": binding["payload_sha256"],
            "attestation_mac": binding["attestation_mac"],
            "session_nonce": V1_3_1_SUPERSEDED_SESSION_NONCE,
        }
        for kind, binding in (("launch", launch_binding), ("terminal", terminal_binding))
    ]
    expected_session_projection = {
        "actual_session_argv": launch["actual_session_argv"],
        "child_process_returncode": terminal["child_process_returncode"],
        "completed_result_payload_sha256": terminal["completed_result_payload_sha256"],
        "completed_work_payload_sha256": terminal["completed_work_payload_sha256"],
        "coordinate_count": cast(Mapping[str, Any], plan)["coordinate_count"],
        "coordinate_digest": cast(Mapping[str, Any], plan)["coordinate_digest"],
        "launch_authority_nonce": launch["launch_authority_nonce"],
        "max_new_cells_stop_limit": cast(Mapping[str, Any], plan)[
            "max_new_cells_stop_limit"
        ],
        "plan_payload_sha256": cast(Mapping[str, Any], plan)["payload_sha256"],
        "published_bundle_reingestion_count": terminal[
            "published_bundle_reingestion_count"
        ],
        "ready_model_load_observed": False,
        "scale": cast(Mapping[str, Any], plan)["scale"],
        "session_nonce": launch["session_nonce"],
        "status": terminal["status"],
        "training_seed": cast(Mapping[str, Any], plan)["training_seed"],
        "worker_count": cast(Mapping[str, Any], plan)["worker_count"],
        "worker_index": cast(Mapping[str, Any], plan)["worker_index"],
    }
    expected_session_counters = {
        "launch_attempt_count": 1,
        "ready_model_load_count": 0,
        "observed_successful_model_loads": 0,
        "terminal_count": 1,
        "published_bundle_reingestion_count": 0,
        "launch_authority_count": 1,
        "controlled_stop_session_count": 1,
    }
    persistent_session_ledger = matrix.get("persistent_session_ledger")
    _require(
        isinstance(persistent_session_ledger, Mapping),
        "Superseded v1.3.1 matrix persistent-session projection is missing.",
    )
    persistent_session_ledger_map = cast(Mapping[str, Any], persistent_session_ledger)
    matrix_session_root = persistent_session_ledger_map.get("root")
    _require(
        matrix_session_root == str(session_root)
        and persistent_session_ledger_map.get("registry") == expected_session_registry
        and persistent_session_ledger_map.get("registry_digest")
        == _json_digest(expected_session_registry)
        and persistent_session_ledger_map.get("sessions") == [expected_session_projection]
        and persistent_session_ledger_map.get("sessions_digest")
        == _json_digest([expected_session_projection])
        and all(
            persistent_session_ledger_map.get(field) == value
            for field, value in expected_session_counters.items()
        )
        and session_lock_path == Path(f"{matrix_session_root}.lock")
        and session_lock_binding["path"] == str(session_lock_path),
        "Superseded v1.3.1 matrix session ledger is not the exact launch/terminal/lock projection.",
    )
    relational_projection = _validate_v1_3_1_failure_relations(
        manifest_public_binding=manifest_public_binding,
        admission_public_binding=admission_public_binding,
        genesis_public_binding=genesis_public_binding,
        activation_public_binding=activation_public_binding,
        inherited_public_binding=inherited.public_binding,
        activation=activation,
        matrix=matrix,
        launch=launch,
        terminal=terminal,
        launch_binding=launch_binding,
        terminal_binding=terminal_binding,
        session_root=session_root,
        session_lock_path=session_lock_path,
        session_lock_binding=session_lock_binding,
    )
    _require(
        relational_projection["registry"] == expected_session_registry
        and relational_projection["session"] == expected_session_projection
        and relational_projection["counters"] == expected_session_counters,
        "Superseded v1.3.1 relational projection disagrees with its authenticated loader.",
    )

    absent_paths = tuple(
        _absolute(path, repository_root=root)
        for path in (
            V1_3_1_SUPERSEDED_WORKER_ROOT,
            V1_3_1_SUPERSEDED_INTEGRITY_PATH,
            V1_3_1_SUPERSEDED_SUMMARY_PATH,
        )
    )
    for path in absent_paths:
        _require(not os.path.lexists(path), f"Forbidden v1.3.1 post-quality artifact exists: {path}")

    source = {
        "schema_version": 1,
        "lineage_type": "signed-superseded-zero-quality-launch-failure",
        "experiment_id": V1_3_1_SUPERSEDED_EXPERIMENT_ID,
        "result_source_commit": V1_3_1_SUPERSEDED_RESULT_SOURCE_COMMIT,
        "result_source_tree": V1_3_1_SUPERSEDED_RESULT_SOURCE_TREE,
        "implementation": {
            "source_commit": V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT,
            "source_tree": V1_3_1_SUPERSEDED_IMPLEMENTATION_SOURCE_TREE,
            "tree_digest": V1_3_1_SUPERSEDED_IMPLEMENTATION_DIGEST,
        },
        "manifest": manifest_binding,
        "reuse_admission": admission_binding,
        "preheldout_genesis": genesis_binding,
        "cross_artifact_public_bindings": {
            "manifest": manifest_public_binding,
            "reuse_admission": admission_public_binding,
            "preheldout_genesis": genesis_public_binding,
            "quality_start_activation": activation_public_binding,
        },
        "activation_root": {
            **activation_inventory,
            "matrix_lock": activation_lock_binding,
            "quality_start_activation": activation_binding,
        },
        "output_closed_world": {
            **output_inventory,
            "matrix": matrix_binding,
            "orphan_claim": claim_binding,
        },
        "persistent_session": {
            **session_inventory,
            "lock": session_lock_binding,
            "launch": launch_binding,
            "terminal": terminal_binding,
            "session_nonce": V1_3_1_SUPERSEDED_SESSION_NONCE,
            "launch_authority_nonce": V1_3_1_SUPERSEDED_LAUNCH_AUTHORITY_NONCE,
            "terminal_status": "launch_failure",
            "child_process_returncode": 1,
            "matrix_projection": {
                "root": str(session_root),
                "session_lock_sibling_path": str(session_lock_path),
                "registry_digest": relational_projection["registry_digest"],
                "sessions_digest": relational_projection["sessions_digest"],
                **expected_session_counters,
            },
        },
        "absent_paths": [str(path) for path in absent_paths],
        "inherited_v1_3_empty_lineage_sha256": V1_3_SUPERSEDED_LINEAGE_SHA256,
        "attestation_key_id": trust_root.key_id,
        "quality_state": _v1_3_1_zero_quality_state(),
    }
    normalized_projection = _require_v1_3_1_normalized_failure_projection(
        _v1_3_1_normalized_failure_projection(source, repository_root=root)
    )
    source["normalized_contract_projection"] = normalized_projection
    source["normalized_contract_projection_sha256"] = normalized_projection[
        "projection_sha256"
    ]
    _v1_3_1_revalidate_regular_binding(
        manifest_binding,
        label="Superseded v1.3.1 manifest",
    )
    _v1_3_1_revalidate_attested_binding(
        admission_binding,
        admission,
        trust_root=trust_root,
        label="Superseded v1.3.1 reuse admission",
    )
    _v1_3_1_revalidate_attested_binding(
        genesis_binding,
        genesis,
        trust_root=trust_root,
        label="Superseded v1.3.1 pre-heldout genesis",
    )
    _v1_3_1_revalidate_regular_binding(
        activation_lock_binding,
        label="Superseded v1.3.1 activation lock",
    )
    _v1_3_1_revalidate_attested_binding(
        activation_binding,
        activation,
        trust_root=trust_root,
        label="Superseded v1.3.1 quality-start activation",
    )
    _v1_3_1_revalidate_attested_binding(
        matrix_binding,
        matrix,
        trust_root=trust_root,
        label="Superseded v1.3.1 matrix",
    )
    _v1_3_1_revalidate_regular_binding(
        claim_binding,
        label="Superseded v1.3.1 orphan claim",
    )
    _v1_3_1_revalidate_regular_binding(
        session_lock_binding,
        label="Superseded v1.3.1 session lock",
    )
    _v1_3_1_revalidate_attested_binding(
        launch_binding,
        launch,
        trust_root=trust_root,
        label="Superseded v1.3.1 session launch",
    )
    _v1_3_1_revalidate_attested_binding(
        terminal_binding,
        terminal,
        trust_root=trust_root,
        label="Superseded v1.3.1 session terminal",
    )
    _v1_3_1_revalidate_closed_world_inventory(
        admission_inventory,
        expected_directories=admission_expected_directories,
        expected_files=admission_expected_files,
        label="Superseded v1.3.1 admission root",
    )
    _v1_3_1_revalidate_closed_world_inventory(
        activation_inventory,
        expected_directories=activation_expected_directories,
        expected_files=activation_expected_files,
        label="Superseded v1.3.1 activation root",
    )
    _v1_3_1_revalidate_closed_world_inventory(
        output_inventory,
        expected_directories=output_expected_directories,
        expected_files=output_expected_files,
        label="Superseded v1.3.1 output root",
    )
    _v1_3_1_revalidate_closed_world_inventory(
        session_inventory,
        expected_directories=session_expected_directories,
        expected_files=session_expected_files,
        label="Superseded v1.3.1 persistent-session root",
    )
    _v1_3_1_require_absent_paths_and_dead_owner(
        absent_paths=absent_paths,
        pid=V1_3_1_SUPERSEDED_CLAIM_PID,
        process_start_ticks=V1_3_1_SUPERSEDED_CLAIM_PROCESS_START_TICKS,
        boundary="immediately before capability return",
    )
    public_binding = {**source, "lineage_sha256": _json_digest(source)}
    return SupersededZeroQualityFailureLineageV1_3_1(
        _seal=_SUPERSEDED_ZERO_QUALITY_FAILURE_LINEAGE_V1_3_1_SEAL,
        manifest=manifest,
        reuse_admission=admission,
        preheldout_genesis=genesis,
        quality_start_activation=activation,
        matrix=matrix,
        persistent_launch=launch,
        persistent_terminal=terminal,
        orphan_claim=claim,
        public_binding=public_binding,
    )


def _require_v1_3_2_context(
    quality_context: QualityContext, *, trust_root: attestation.TrustRoot
) -> None:
    _require(type(quality_context) is QualityContext, "Quality context must be canonical.")
    _require(
        quality_context.manifest_binding.get("experiment_id") == V1_3_2_QUALITY_EXPERIMENT_ID
        and quality_context.manifest_binding.get("attestation", {}).get("key_id")
        == trust_root.key_id
        == V1_3_SUPERSEDED_ATTESTATION_KEY_ID,
        "Quality context is not the canonical v1.3.2 trust boundary.",
    )
    assert_quality_context_unchanged(quality_context)


def _require_superseded_lineage(
    lineage: SupersededEmptyLineageV1_3,
) -> SupersededEmptyLineageV1_3:
    _require(
        type(lineage) is SupersededEmptyLineageV1_3
        and lineage._seal is _SUPERSEDED_EMPTY_LINEAGE_SEAL
        and lineage.public_binding.get("lineage_sha256")
        == _json_digest(
            {key: value for key, value in lineage.public_binding.items() if key != "lineage_sha256"}
        ),
        "Superseded-empty lineage capability is invalid.",
    )
    return lineage


def _require_superseded_failure_lineage(
    lineage: SupersededZeroQualityFailureLineageV1_3_1,
) -> SupersededZeroQualityFailureLineageV1_3_1:
    _require(
        type(lineage) is SupersededZeroQualityFailureLineageV1_3_1
        and lineage._seal is _SUPERSEDED_ZERO_QUALITY_FAILURE_LINEAGE_V1_3_1_SEAL
        and lineage.public_binding.get("lineage_sha256")
        == _json_digest(
            {key: value for key, value in lineage.public_binding.items() if key != "lineage_sha256"}
        ),
        "Superseded v1.3.1 failure-lineage capability is invalid.",
    )
    projection = lineage.public_binding.get("normalized_contract_projection")
    projection_sha256 = lineage.public_binding.get(
        "normalized_contract_projection_sha256"
    )
    _require(
        isinstance(projection, Mapping)
        and _is_sha256(projection_sha256)
        and projection_sha256 == cast(Mapping[str, Any], projection).get("projection_sha256"),
        "Superseded v1.3.1 normalized failure-lineage capability is invalid.",
    )
    _require_v1_3_1_normalized_failure_projection(cast(Mapping[str, Any], projection))
    return lineage


def _superseded_failure_projection_sha256(
    lineage: SupersededZeroQualityFailureLineageV1_3_1,
) -> str:
    checked = _require_superseded_failure_lineage(lineage)
    return cast(str, checked.public_binding["normalized_contract_projection_sha256"])


def _v1_3_2_prestart_absent_paths(*, repository_root: Path) -> tuple[str, ...]:
    return (
        *_assert_v1_3_2_quality_paths_absent(repository_root=repository_root),
        _assert_v1_3_2_activation_root_absent(repository_root=repository_root),
    )


def _v1_3_2_reuse_admission_public_binding(
    *, path: Path, payload: Mapping[str, Any], sha256: str, byte_count: int
) -> dict[str, Any]:
    envelope = payload.get("attestation")
    lineage = payload.get("superseded_empty_lineage")
    failure_lineage = payload.get("superseded_zero_quality_failure_lineage")
    _require(
        isinstance(envelope, Mapping)
        and isinstance(lineage, Mapping)
        and isinstance(failure_lineage, Mapping)
        and payload.get("superseded_zero_quality_failure_lineage_projection_sha256")
        == cast(Mapping[str, Any], failure_lineage).get(
            "normalized_contract_projection_sha256"
        ),
        "V1.3.2 reuse-admission binding inputs are invalid.",
    )
    lineage_map = cast(Mapping[str, Any], lineage)
    old_admission = lineage_map.get("reuse_admission")
    _require(isinstance(old_admission, Mapping), "Superseded admission binding is missing.")
    return {
        "path": str(path),
        "sha256": sha256,
        "bytes": byte_count,
        "experiment_id": payload.get("experiment_id"),
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": cast(Mapping[str, Any], envelope).get("mac"),
        "historical_receipt_sha256": lineage_map.get("historical_receipt_payload_sha256"),
        "canonical_nonobservation_sha256": lineage_map.get(
            "canonical_nonobservation_payload_sha256"
        ),
        "superseded_failure_lineage_sha256": cast(Mapping[str, Any], failure_lineage).get(
            "lineage_sha256"
        ),
        "superseded_failure_lineage_projection_sha256": payload.get(
            "superseded_zero_quality_failure_lineage_projection_sha256"
        ),
    }


def build_v1_3_2_reuse_admission_payload(
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    superseded_empty_lineage: SupersededEmptyLineageV1_3,
    superseded_failure_lineage: SupersededZeroQualityFailureLineageV1_3_1,
    admission_nonce: str | None = None,
) -> dict[str, Any]:
    _require_v1_3_2_context(quality_context, trust_root=trust_root)
    lineage = _require_superseded_lineage(superseded_empty_lineage)
    failure_lineage = _require_superseded_failure_lineage(superseded_failure_lineage)
    failure_projection_sha256 = _superseded_failure_projection_sha256(failure_lineage)
    nonce = secrets.token_hex(32) if admission_nonce is None else admission_nonce
    _require(_is_sha256(nonce), "V1.3.2 admission nonce must contain 256 random bits.")
    path = _absolute(
        V1_3_2_DEFAULT_ADMISSION_PATH,
        repository_root=quality_context.repository_root,
    )
    _require(
        not os.path.lexists(path),
        "V1.3.2 reuse-admission creation refuses an existing final path.",
    )
    absent = _v1_3_2_prestart_absent_paths(repository_root=quality_context.repository_root)
    old_projection = lineage.reuse_admission.get("execution_environment_projection")
    _require(isinstance(old_projection, Mapping), "Superseded execution projection is missing.")
    return _attested_payload(
        {
            "schema_version": V1_3_2_ADMISSION_SCHEMA_VERSION,
            "artifact_type": "direct-controller-v1-3-2-historical-reuse-admission",
            "experiment_id": V1_3_2_QUALITY_EXPERIMENT_ID,
            "status": "terminal",
            "canonical_path": str(path),
            "admission_nonce": nonce,
            "quality_source": quality_context.source,
            "quality_manifest": quality_context.manifest_binding,
            "superseded_empty_lineage": lineage.public_binding,
            "superseded_zero_quality_failure_lineage": failure_lineage.public_binding,
            "superseded_zero_quality_failure_lineage_projection_sha256": (
                failure_projection_sha256
            ),
            "prospective_quality_absent_paths": list(absent[:-1]),
            "activation_root_absent_path": absent[-1],
            "execution_environment_projection": dict(cast(Mapping[str, Any], old_projection)),
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
            "active_claim_count": 0,
            "worker_ledger_count": 0,
            "top_p_quality_input_count": 0,
            "historical_artifacts_rewritten_or_reattested": False,
            "scientific_subprocesses_started_during_admission": 0,
        },
        trust_root=trust_root,
        purpose=V1_3_2_REUSE_ADMISSION_PURPOSE,
    )


def _v1_3_2_admission_entries(
    lineage: SupersededEmptyLineageV1_3, *, quality_context: QualityContext
) -> tuple[
    dict[tuple[str, int], AdmittedCalibration],
    dict[tuple[str, int], AdmittedCheckpoint],
]:
    return _admission_entries(lineage.reuse_admission, quality_context=quality_context)


def _validate_v1_3_2_reuse_admission_internal(
    payload: Mapping[str, Any],
    *,
    admission_path: Path,
    storage_path: Path,
    require_final_root: bool,
    require_prestart_absence: bool,
    trust_root: attestation.TrustRoot,
    quality_context: QualityContext,
    consumer_scope: _ActivatedConsumerScopeV1_3_2 | None = None,
) -> ValidatedReuseAdmission:
    _require_v1_3_2_context(quality_context, trust_root=trust_root)
    lineage = load_superseded_empty_lineage_v1_3(
        trust_root=trust_root,
        repository_root=quality_context.repository_root,
    )
    failure_lineage = load_superseded_zero_quality_failure_lineage_v1_3_1(
        trust_root=trust_root,
        repository_root=quality_context.repository_root,
        superseded_empty_lineage=lineage,
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
        "superseded_empty_lineage",
        "superseded_zero_quality_failure_lineage",
        "superseded_zero_quality_failure_lineage_projection_sha256",
        "prospective_quality_absent_paths",
        "activation_root_absent_path",
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
        "active_claim_count",
        "worker_ledger_count",
        "top_p_quality_input_count",
        "historical_artifacts_rewritten_or_reattested",
        "scientific_subprocesses_started_during_admission",
        "payload_sha256",
        "attestation",
    }
    _require(set(payload) == expected_fields, "V1.3.2 reuse-admission schema drifted.")
    _verify_attested_payload(
        payload,
        trust_root=trust_root,
        purpose=V1_3_2_REUSE_ADMISSION_PURPOSE,
        label="V1.3.2 reuse admission",
    )
    canonical_path = _absolute(
        V1_3_2_DEFAULT_ADMISSION_PATH,
        repository_root=quality_context.repository_root,
    )
    path = _exact_path(
        admission_path,
        label="Canonical v1.3.2 reuse admission",
        repository_root=quality_context.repository_root,
        must_exist=require_final_root,
    )
    if require_final_root:
        final_admission, _final_genesis = _validate_admission_bundle_root(
            _absolute(V1_3_2_ADMISSION_ROOT, repository_root=quality_context.repository_root),
            label="Final v1.3.2 admission root",
        )
        _require(path == final_admission, "V1.3.2 reuse-admission final path drifted.")
    old_projection = lineage.reuse_admission.get("execution_environment_projection")
    zero_fields = (
        "evaluation_inputs_materialized",
        "quality_predictions_materialized",
        "quality_outcomes_materialized",
        "quality_aggregates_materialized",
        "active_claim_count",
        "worker_ledger_count",
        "top_p_quality_input_count",
        "scientific_subprocesses_started_during_admission",
    )
    expected_absent = tuple(
        str(_absolute(item, repository_root=quality_context.repository_root))
        for item in V1_3_2_PROSPECTIVE_QUALITY_PATHS
    )
    expected_activation = str(
        _absolute(V1_3_2_ACTIVATION_ROOT, repository_root=quality_context.repository_root)
    )
    _require(
        path == canonical_path
        and payload.get("schema_version") == V1_3_2_ADMISSION_SCHEMA_VERSION
        and payload.get("artifact_type") == "direct-controller-v1-3-2-historical-reuse-admission"
        and payload.get("experiment_id") == V1_3_2_QUALITY_EXPERIMENT_ID
        and payload.get("status") == "terminal"
        and payload.get("canonical_path") == str(canonical_path)
        and _is_sha256(payload.get("admission_nonce"))
        and payload.get("quality_source") == quality_context.source
        and payload.get("quality_manifest") == quality_context.manifest_binding
        and payload.get("superseded_empty_lineage") == lineage.public_binding
        and payload.get("superseded_zero_quality_failure_lineage")
        == failure_lineage.public_binding
        and payload.get("superseded_zero_quality_failure_lineage_projection_sha256")
        == _superseded_failure_projection_sha256(failure_lineage)
        and tuple(payload.get("prospective_quality_absent_paths", ())) == expected_absent
        and payload.get("activation_root_absent_path") == expected_activation
        and isinstance(old_projection, Mapping)
        and payload.get("execution_environment_projection") == dict(old_projection)
        and tuple(payload.get("training_seeds", ())) == TRAINING_SEEDS
        and tuple(payload.get("calibration_seeds", ())) == CALIBRATION_SEEDS
        and tuple(payload.get("evaluation_seeds", ())) == EVALUATION_SEEDS
        and payload.get("evaluation_seed_namespace_reselected") is False
        and payload.get("quality_evaluation_started") is False
        and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and payload.get("quality_rng_initialized") is False
        and all(payload.get(field) == 0 for field in zero_fields)
        and payload.get("historical_artifacts_rewritten_or_reattested") is False,
        "V1.3.2 reuse-admission scientific or lineage contract drifted.",
    )
    if require_prestart_absence:
        _v1_3_2_prestart_absent_paths(repository_root=quality_context.repository_root)
    storage = _exact_path(
        storage_path,
        label="V1.3.2 reuse-admission storage",
        repository_root=quality_context.repository_root,
        must_exist=True,
    )
    disk, opened = _load_json_nofollow(
        storage,
        label="V1.3.2 reuse admission",
        require_canonical_pretty_bytes=True,
    )
    try:
        _require(disk == dict(payload), "V1.3.2 admission differs from exact disk bytes.")
        public_binding = _v1_3_2_reuse_admission_public_binding(
            path=canonical_path,
            payload=payload,
            sha256=opened.sha256,
            byte_count=opened.bytes,
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    calibrations, checkpoints = _v1_3_2_admission_entries(
        lineage,
        quality_context=quality_context,
    )
    result = ValidatedReuseAdmission(
        payload=dict(payload),
        public_binding=public_binding,
        calibrations=calibrations,
        checkpoints=checkpoints,
        quality_context=quality_context,
        execution_environment_projection=dict(cast(Mapping[str, Any], old_projection)),
    )
    coordinates_to_validate: tuple[tuple[str, int], ...]
    if consumer_scope is None:
        coordinates_to_validate = tuple(sorted(calibrations))
    else:
        _require(
            type(consumer_scope) is _ActivatedConsumerScopeV1_3_2
            and consumer_scope._seal is _ACTIVATED_CONSUMER_SCOPE_V1_3_2_SEAL
            and consumer_scope.coordinate in calibrations
            and consumer_scope.coordinate in checkpoints,
            "Activated consumer scope is invalid or unregistered.",
        )
        admitted_calibration = calibrations[consumer_scope.coordinate].public_binding
        admitted_checkpoint = checkpoints[consumer_scope.coordinate].public_binding
        _require(
            set(consumer_scope.calibration_binding)
            == {
                "path",
                "sha256",
                "bytes",
                "payload_sha256",
                "attestation_mac",
                "experiment_id",
            }
            and all(
                admitted_calibration.get(field) == value
                for field, value in consumer_scope.calibration_binding.items()
            )
            and consumer_scope.checkpoint_binding == admitted_checkpoint,
            "Activated consumer bindings differ from the admitted coordinate.",
        )
        coordinates_to_validate = (consumer_scope.coordinate,)
    for coordinate in coordinates_to_validate:
        calibration_payload, calibration_opened = _load_json_nofollow(
            calibrations[coordinate].path,
            label="V1.3.2 inherited admitted calibration",
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
            label="V1.3.2 inherited admitted checkpoint",
            repository_root=quality_context.repository_root,
        )
    if consumer_scope is not None:
        coordinate = consumer_scope.coordinate
        result = ValidatedReuseAdmission(
            payload=result.payload,
            public_binding=result.public_binding,
            calibrations={coordinate: calibrations[coordinate]},
            checkpoints={coordinate: checkpoints[coordinate]},
            quality_context=result.quality_context,
            execution_environment_projection=result.execution_environment_projection,
        )
    assert_quality_context_unchanged(quality_context)
    return result


def _v1_3_2_genesis_public_binding(
    *, path: Path, payload: Mapping[str, Any], sha256: str, byte_count: int
) -> dict[str, Any]:
    envelope = payload.get("attestation")
    reuse_admission = payload.get("reuse_admission")
    failure_lineage = payload.get("superseded_zero_quality_failure_lineage")
    _require(
        isinstance(envelope, Mapping)
        and isinstance(reuse_admission, Mapping)
        and isinstance(failure_lineage, Mapping)
        and payload.get("superseded_zero_quality_failure_lineage_projection_sha256")
        == cast(Mapping[str, Any], failure_lineage).get(
            "normalized_contract_projection_sha256"
        ),
        "V1.3.2 genesis binding inputs are invalid.",
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
        "superseded_failure_lineage_sha256": cast(Mapping[str, Any], failure_lineage).get(
            "lineage_sha256"
        ),
        "superseded_failure_lineage_projection_sha256": payload.get(
            "superseded_zero_quality_failure_lineage_projection_sha256"
        ),
    }


def build_v1_3_2_preheldout_genesis_payload(
    *,
    admission: ValidatedReuseAdmission,
    trust_root: attestation.TrustRoot,
    superseded_empty_lineage: SupersededEmptyLineageV1_3,
    superseded_failure_lineage: SupersededZeroQualityFailureLineageV1_3_1,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> dict[str, Any]:
    _require(type(admission) is ValidatedReuseAdmission, "V1.3.2 genesis requires admission.")
    _require_v1_3_2_context(admission.quality_context, trust_root=trust_root)
    lineage = _require_superseded_lineage(superseded_empty_lineage)
    failure_lineage = _require_superseded_failure_lineage(superseded_failure_lineage)
    failure_projection_sha256 = _superseded_failure_projection_sha256(failure_lineage)
    arms = _validate_quality_genesis_registration(
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    path = _absolute(
        V1_3_2_DEFAULT_GENESIS_PATH,
        repository_root=admission.quality_context.repository_root,
    )
    _require(not os.path.lexists(path), "V1.3.2 genesis refuses an existing final path.")
    _v1_3_2_prestart_absent_paths(repository_root=admission.quality_context.repository_root)
    return _attested_payload(
        {
            "schema_version": V1_3_2_GENESIS_SCHEMA_VERSION,
            "artifact_type": "direct-controller-v1-3-2-preheldout-genesis",
            "experiment_id": V1_3_2_QUALITY_EXPERIMENT_ID,
            "status": "in_progress",
            "canonical_path": str(path),
            "quality_source": admission.quality_context.source,
            "quality_manifest": admission.quality_context.manifest_binding,
            "reuse_admission": admission.public_binding,
            "superseded_empty_lineage": lineage.public_binding,
            "superseded_zero_quality_failure_lineage": failure_lineage.public_binding,
            "superseded_zero_quality_failure_lineage_projection_sha256": (
                failure_projection_sha256
            ),
            "activation_root": str(
                _absolute(
                    V1_3_2_ACTIVATION_ROOT,
                    repository_root=admission.quality_context.repository_root,
                )
            ),
            "activation_matrix_lock_path": str(
                _absolute(
                    V1_3_2_ACTIVATION_MATRIX_LOCK_PATH,
                    repository_root=admission.quality_context.repository_root,
                )
            ),
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
        purpose=V1_3_2_PREHELDOUT_GENESIS_PURPOSE,
    )


def _validate_v1_3_2_preheldout_genesis_internal(
    payload: Mapping[str, Any],
    *,
    genesis_path: Path,
    storage_path: Path,
    require_final_root: bool,
    require_prestart_absence: bool,
    admission: ValidatedReuseAdmission,
    superseded_empty_lineage: SupersededEmptyLineageV1_3,
    superseded_failure_lineage: SupersededZeroQualityFailureLineageV1_3_1,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> ValidatedPreheldoutGenesis:
    _require(type(admission) is ValidatedReuseAdmission, "V1.3.2 genesis requires admission.")
    _require_v1_3_2_context(admission.quality_context, trust_root=trust_root)
    lineage = _require_superseded_lineage(superseded_empty_lineage)
    failure_lineage = _require_superseded_failure_lineage(superseded_failure_lineage)
    arms = _validate_quality_genesis_registration(
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
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
        "superseded_empty_lineage",
        "superseded_zero_quality_failure_lineage",
        "superseded_zero_quality_failure_lineage_projection_sha256",
        "activation_root",
        "activation_matrix_lock_path",
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
    _require(set(payload) == expected_fields, "V1.3.2 genesis schema drifted.")
    _verify_attested_payload(
        payload,
        trust_root=trust_root,
        purpose=V1_3_2_PREHELDOUT_GENESIS_PURPOSE,
        label="V1.3.2 pre-heldout genesis",
    )
    canonical_path = _absolute(
        V1_3_2_DEFAULT_GENESIS_PATH,
        repository_root=admission.quality_context.repository_root,
    )
    path = _exact_path(
        genesis_path,
        label="Canonical v1.3.2 pre-heldout genesis",
        repository_root=admission.quality_context.repository_root,
        must_exist=require_final_root,
    )
    if require_final_root:
        _final_admission, final_genesis = _validate_admission_bundle_root(
            _absolute(
                V1_3_2_ADMISSION_ROOT,
                repository_root=admission.quality_context.repository_root,
            ),
            label="Final v1.3.2 admission root",
        )
        _require(path == final_genesis, "V1.3.2 genesis final path drifted.")
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
    root = admission.quality_context.repository_root
    _require(
        path == canonical_path
        and payload.get("schema_version") == V1_3_2_GENESIS_SCHEMA_VERSION
        and payload.get("artifact_type") == "direct-controller-v1-3-2-preheldout-genesis"
        and payload.get("experiment_id") == V1_3_2_QUALITY_EXPERIMENT_ID
        and payload.get("status") == "in_progress"
        and payload.get("canonical_path") == str(canonical_path)
        and payload.get("quality_source") == admission.quality_context.source
        and payload.get("quality_manifest") == admission.quality_context.manifest_binding
        and payload.get("reuse_admission") == admission.public_binding
        and payload.get("superseded_empty_lineage") == lineage.public_binding
        and payload.get("superseded_zero_quality_failure_lineage")
        == failure_lineage.public_binding
        and payload.get("superseded_zero_quality_failure_lineage_projection_sha256")
        == _superseded_failure_projection_sha256(failure_lineage)
        and payload.get("activation_root")
        == str(_absolute(V1_3_2_ACTIVATION_ROOT, repository_root=root))
        and payload.get("activation_matrix_lock_path")
        == str(_absolute(V1_3_2_ACTIVATION_MATRIX_LOCK_PATH, repository_root=root))
        and payload.get("expected_shards") == expected_shards
        and payload.get("coordinate_digest") == coordinate_digest
        and tuple(payload.get("exact_fill_arm_names", ())) == arms
        and payload.get("records") == []
        and payload.get("quality_evaluation_started") is False
        and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
        and payload.get("quality_rng_initialized") is False
        and all(payload.get(field) == 0 for field in zero_fields),
        "V1.3.2 genesis zero-prefix or lineage contract drifted.",
    )
    if require_prestart_absence:
        _v1_3_2_prestart_absent_paths(repository_root=root)
    storage = _exact_path(
        storage_path,
        label="V1.3.2 genesis storage",
        repository_root=root,
        must_exist=True,
    )
    disk, opened = _load_json_nofollow(
        storage,
        label="V1.3.2 pre-heldout genesis",
        require_canonical_pretty_bytes=True,
    )
    try:
        _require(disk == dict(payload), "V1.3.2 genesis differs from exact disk bytes.")
        public_binding = _v1_3_2_genesis_public_binding(
            path=canonical_path,
            payload=payload,
            sha256=opened.sha256,
            byte_count=opened.bytes,
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    assert_quality_context_unchanged(admission.quality_context)
    return ValidatedPreheldoutGenesis(payload=dict(payload), public_binding=public_binding)


def _recover_v1_3_2_admission_staging(parent: Path) -> None:
    recovered = False
    with os.scandir(parent) as iterator:
        candidates = sorted(
            Path(entry.path)
            for entry in iterator
            if entry.name.startswith(V1_3_2_ADMISSION_STAGING_PREFIX)
        )
    for candidate in candidates:
        _remove_safe_staging_directory(candidate)
        recovered = True
    if recovered:
        _fsync_directory(parent)


def publish_v1_3_2_admission_genesis_bundle(
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> tuple[ValidatedReuseAdmission, ValidatedPreheldoutGenesis]:
    """Publish the new static pair; never mutate or relabel the v1.3 pair."""

    _require_v1_3_2_context(quality_context, trust_root=trust_root)
    arms = _validate_quality_genesis_registration(
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    lineage = load_superseded_empty_lineage_v1_3(
        trust_root=trust_root,
        repository_root=quality_context.repository_root,
    )
    failure_lineage = load_superseded_zero_quality_failure_lineage_v1_3_1(
        trust_root=trust_root,
        repository_root=quality_context.repository_root,
        superseded_empty_lineage=lineage,
    )
    root = _absolute(V1_3_2_ADMISSION_ROOT, repository_root=quality_context.repository_root)
    parent = _exact_path(root.parent, label="V1.3.2 admission parent", must_exist=True)
    _v1_3_2_prestart_absent_paths(repository_root=quality_context.repository_root)
    from adaptive_v4_gpu_lock import acquire_gpu_lock

    bootstrap = acquire_gpu_lock(
        "p2-direct-controller-v1.3.2-admission-publication",
        path=DIRECT_GPU_SCHEDULER_LOCK_PATH,
    )
    staging: Path | None = None
    published = False
    try:
        bootstrap.assert_held()
        _fsync_directory(parent)
        _require(not os.path.lexists(root), "V1.3.2 final admission root is immutable.")
        _recover_v1_3_2_admission_staging(parent)
        _require(not os.path.lexists(root), "V1.3.2 admission root appeared during recovery.")
        _v1_3_2_prestart_absent_paths(repository_root=quality_context.repository_root)
        admission_payload = build_v1_3_2_reuse_admission_payload(
            quality_context=quality_context,
            trust_root=trust_root,
            superseded_empty_lineage=lineage,
            superseded_failure_lineage=failure_lineage,
        )
        admission_bytes = canonical_pretty_json(admission_payload)
        admission_binding = _v1_3_2_reuse_admission_public_binding(
            path=_absolute(
                V1_3_2_DEFAULT_ADMISSION_PATH,
                repository_root=quality_context.repository_root,
            ),
            payload=admission_payload,
            sha256=hashlib.sha256(admission_bytes).hexdigest(),
            byte_count=len(admission_bytes),
        )
        calibrations, checkpoints = _v1_3_2_admission_entries(
            lineage,
            quality_context=quality_context,
        )
        provisional = ValidatedReuseAdmission(
            payload=dict(admission_payload),
            public_binding=admission_binding,
            calibrations=calibrations,
            checkpoints=checkpoints,
            quality_context=quality_context,
            execution_environment_projection=dict(
                cast(Mapping[str, Any], admission_payload["execution_environment_projection"])
            ),
        )
        genesis_payload = build_v1_3_2_preheldout_genesis_payload(
            admission=provisional,
            trust_root=trust_root,
            superseded_empty_lineage=lineage,
            superseded_failure_lineage=failure_lineage,
            expected_shards=expected_shards,
            coordinate_digest=coordinate_digest,
            exact_fill_arm_names=arms,
        )
        genesis_bytes = canonical_pretty_json(genesis_payload)
        staging = parent / f"{V1_3_2_ADMISSION_STAGING_PREFIX}{secrets.token_hex(16)}"
        os.mkdir(staging, SAFE_DIRECTORY_MODE)
        os.chmod(staging, SAFE_DIRECTORY_MODE)
        _write_exclusive_durable(staging / V1_3_2_DEFAULT_ADMISSION_PATH.name, admission_bytes)
        _write_exclusive_durable(staging / V1_3_2_DEFAULT_GENESIS_PATH.name, genesis_bytes)
        _fsync_directory(staging, exact_mode=SAFE_DIRECTORY_MODE)
        _validate_admission_bundle_root(staging, label="Staged v1.3.2 admission root")
        staged_admission = _validate_v1_3_2_reuse_admission_internal(
            admission_payload,
            admission_path=root / V1_3_2_DEFAULT_ADMISSION_PATH.name,
            storage_path=staging / V1_3_2_DEFAULT_ADMISSION_PATH.name,
            require_final_root=False,
            require_prestart_absence=True,
            trust_root=trust_root,
            quality_context=quality_context,
        )
        _require(
            staged_admission.public_binding == admission_binding,
            "Staged v1.3.2 admission differs from its provisional binding.",
        )
        staged_genesis = _validate_v1_3_2_preheldout_genesis_internal(
            genesis_payload,
            genesis_path=root / V1_3_2_DEFAULT_GENESIS_PATH.name,
            storage_path=staging / V1_3_2_DEFAULT_GENESIS_PATH.name,
            require_final_root=False,
            require_prestart_absence=True,
            admission=staged_admission,
            superseded_empty_lineage=lineage,
            superseded_failure_lineage=failure_lineage,
            trust_root=trust_root,
            expected_shards=expected_shards,
            coordinate_digest=coordinate_digest,
            exact_fill_arm_names=arms,
        )
        bootstrap.assert_held()
        _v1_3_2_prestart_absent_paths(repository_root=quality_context.repository_root)
        _require(not os.path.lexists(root), "V1.3.2 admission root appeared before publish.")
        _rename_directory_noreplace(staging, root)
        published = True
        _fsync_directory(parent)
        _validate_admission_bundle_root(root, label="Final v1.3.2 admission root")
        _validate_file_binding(
            staged_admission.public_binding,
            label="Published v1.3.2 reuse admission",
            repository_root=quality_context.repository_root,
        )
        _validate_file_binding(
            staged_genesis.public_binding,
            label="Published v1.3.2 genesis",
            repository_root=quality_context.repository_root,
        )
        bootstrap.assert_held()
        return staged_admission, staged_genesis
    finally:
        try:
            if staging is not None and not published and os.path.lexists(staging):
                _remove_safe_staging_directory(staging)
                _fsync_directory(parent)
        finally:
            bootstrap.close()


def _load_v1_3_2_static_bundle(
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    require_prestart_absence: bool,
    consumer_scope: _ActivatedConsumerScopeV1_3_2 | None = None,
) -> tuple[
    ValidatedReuseAdmission,
    ValidatedPreheldoutGenesis,
    SupersededEmptyLineageV1_3,
    SupersededZeroQualityFailureLineageV1_3_1,
]:
    root = _absolute(V1_3_2_ADMISSION_ROOT, repository_root=quality_context.repository_root)
    admission_path, genesis_path = _validate_admission_bundle_root(
        root,
        label="Final v1.3.2 admission root",
    )
    admission_payload, admission_opened = _load_json_nofollow(
        admission_path,
        label="V1.3.2 reuse admission",
        require_canonical_pretty_bytes=True,
    )
    admission_opened.close()
    validated_admission = _validate_v1_3_2_reuse_admission_internal(
        admission_payload,
        admission_path=admission_path,
        storage_path=admission_path,
        require_final_root=True,
        require_prestart_absence=require_prestart_absence,
        trust_root=trust_root,
        quality_context=quality_context,
        consumer_scope=consumer_scope,
    )
    lineage = load_superseded_empty_lineage_v1_3(
        trust_root=trust_root,
        repository_root=quality_context.repository_root,
    )
    failure_lineage = load_superseded_zero_quality_failure_lineage_v1_3_1(
        trust_root=trust_root,
        repository_root=quality_context.repository_root,
        superseded_empty_lineage=lineage,
    )
    genesis_payload, genesis_opened = _load_json_nofollow(
        genesis_path,
        label="V1.3.2 pre-heldout genesis",
        require_canonical_pretty_bytes=True,
    )
    genesis_opened.close()
    validated_genesis = _validate_v1_3_2_preheldout_genesis_internal(
        genesis_payload,
        genesis_path=genesis_path,
        storage_path=genesis_path,
        require_final_root=True,
        require_prestart_absence=require_prestart_absence,
        admission=validated_admission,
        superseded_empty_lineage=lineage,
        superseded_failure_lineage=failure_lineage,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    return validated_admission, validated_genesis, lineage, failure_lineage


def load_prestart_quality_authority(
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> PrestartQualityAuthorityV1_3_2:
    """Create a typed live-absence witness before activation exists."""

    admission, genesis, lineage, failure_lineage = _load_v1_3_2_static_bundle(
        quality_context=quality_context,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
        require_prestart_absence=True,
    )
    absent = _v1_3_2_prestart_absent_paths(repository_root=quality_context.repository_root)
    source = {
        "schema_version": 1,
        "authority_type": "v1.3.2-live-prestart-absence",
        "quality_source": quality_context.source,
        "quality_manifest": quality_context.manifest_binding,
        "reuse_admission": admission.public_binding,
        "preheldout_genesis": genesis.public_binding,
        "superseded_empty_lineage_sha256": lineage.public_binding["lineage_sha256"],
        "superseded_failure_lineage_sha256": failure_lineage.public_binding[
            "lineage_sha256"
        ],
        "absent_paths": list(absent),
        "quality_state": _superseded_empty_quality_state(),
    }
    witness = {**source, "witness_sha256": _json_digest(source)}
    return PrestartQualityAuthorityV1_3_2(
        _seal=_PRESTART_QUALITY_AUTHORITY_V1_3_2_SEAL,
        quality_context=quality_context,
        reuse_admission=admission,
        preheldout_genesis=genesis,
        superseded_empty_lineage=lineage,
        superseded_failure_lineage=failure_lineage,
        absence_witness=witness,
    )


def _validate_prestart_absence_witness(
    value: Mapping[str, Any],
    *,
    quality_context: QualityContext,
    reuse_admission: ValidatedReuseAdmission,
    preheldout_genesis: ValidatedPreheldoutGenesis,
    superseded_empty_lineage: SupersededEmptyLineageV1_3,
    superseded_failure_lineage: SupersededZeroQualityFailureLineageV1_3_1,
) -> dict[str, Any]:
    """Validate the signed historical absence claim without reasserting live absence."""

    expected_absent_paths = [
        str(_absolute(path, repository_root=quality_context.repository_root))
        for path in V1_3_2_PROSPECTIVE_QUALITY_PATHS
    ]
    expected_absent_paths.append(
        str(
            _absolute(
                V1_3_2_ACTIVATION_ROOT,
                repository_root=quality_context.repository_root,
            )
        )
    )
    checked = dict(value)
    source = {key: item for key, item in checked.items() if key != "witness_sha256"}
    _require(
        set(checked)
        == {
            "schema_version",
            "authority_type",
            "quality_source",
            "quality_manifest",
            "reuse_admission",
            "preheldout_genesis",
            "superseded_empty_lineage_sha256",
            "superseded_failure_lineage_sha256",
            "absent_paths",
            "quality_state",
            "witness_sha256",
        }
        and checked.get("schema_version") == 1
        and checked.get("authority_type") == "v1.3.2-live-prestart-absence"
        and checked.get("quality_source") == quality_context.source
        and checked.get("quality_manifest") == quality_context.manifest_binding
        and checked.get("reuse_admission") == reuse_admission.public_binding
        and checked.get("preheldout_genesis") == preheldout_genesis.public_binding
        and checked.get("superseded_empty_lineage_sha256")
        == superseded_empty_lineage.public_binding["lineage_sha256"]
        and checked.get("superseded_failure_lineage_sha256")
        == superseded_failure_lineage.public_binding["lineage_sha256"]
        and checked.get("absent_paths") == expected_absent_paths
        and checked.get("quality_state") == _superseded_empty_quality_state()
        and checked.get("witness_sha256") == _json_digest(source),
        "V1.3.2 prestart absence witness drifted.",
    )
    return checked


def _require_prestart_authority(
    prestart: PrestartQualityAuthorityV1_3_2,
) -> PrestartQualityAuthorityV1_3_2:
    _require(
        type(prestart) is PrestartQualityAuthorityV1_3_2
        and prestart._seal is _PRESTART_QUALITY_AUTHORITY_V1_3_2_SEAL
        and type(prestart.reuse_admission) is ValidatedReuseAdmission
        and type(prestart.preheldout_genesis) is ValidatedPreheldoutGenesis
        and isinstance(prestart.absence_witness, Mapping),
        "Prestart quality authority is raw, stale, or duck-typed.",
    )
    _validate_prestart_absence_witness(
        prestart.absence_witness,
        quality_context=prestart.quality_context,
        reuse_admission=prestart.reuse_admission,
        preheldout_genesis=prestart.preheldout_genesis,
        superseded_empty_lineage=prestart.superseded_empty_lineage,
        superseded_failure_lineage=prestart.superseded_failure_lineage,
    )
    return prestart


def _revalidate_cached_static_bundle(
    authority: PrestartQualityAuthorityV1_3_2 | ValidatedQualityStartActivationV1_3_2,
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    require_live_prestart_absence: bool,
) -> tuple[
    ValidatedReuseAdmission,
    ValidatedPreheldoutGenesis,
    SupersededEmptyLineageV1_3,
    SupersededZeroQualityFailureLineageV1_3_1,
]:
    """Recheck a sealed full-grid capability without rehashing every checkpoint."""

    if type(authority) is PrestartQualityAuthorityV1_3_2:
        checked = _require_prestart_authority(authority)
        cached_context = checked.quality_context
        reuse_admission = checked.reuse_admission
        preheldout_genesis = checked.preheldout_genesis
        lineage = checked.superseded_empty_lineage
        failure_lineage = checked.superseded_failure_lineage
    else:
        checked_activation = _require_full_activation_authority(
            cast(ValidatedQualityStartActivationV1_3_2, authority)
        )
        cached_context = checked_activation.quality_context
        reuse_admission = checked_activation.reuse_admission
        preheldout_genesis = checked_activation.preheldout_genesis
        lineage = checked_activation.superseded_empty_lineage
        failure_lineage = checked_activation.superseded_failure_lineage
    expected_coordinates = {
        (scale, training_seed)
        for scale in SCALES
        for training_seed in TRAINING_SEEDS
    }
    _require(
        cached_context == quality_context
        and reuse_admission.quality_context == quality_context
        and set(reuse_admission.calibrations) == expected_coordinates
        and set(reuse_admission.checkpoints) == expected_coordinates
        and preheldout_genesis.payload.get("expected_shards") == expected_shards
        and preheldout_genesis.payload.get("coordinate_digest") == coordinate_digest
        and tuple(preheldout_genesis.payload.get("exact_fill_arm_names", ()))
        == tuple(exact_fill_arm_names),
        "Cached full-grid static authority registration drifted.",
    )
    _require_v1_3_2_context(quality_context, trust_root=trust_root)
    _verify_attested_payload(
        reuse_admission.payload,
        trust_root=trust_root,
        purpose=V1_3_2_REUSE_ADMISSION_PURPOSE,
        label="Cached v1.3.2 reuse admission",
    )
    _verify_attested_payload(
        preheldout_genesis.payload,
        trust_root=trust_root,
        purpose=V1_3_2_PREHELDOUT_GENESIS_PURPOSE,
        label="Cached v1.3.2 pre-heldout genesis",
    )
    _require_superseded_lineage(lineage)
    _require_superseded_failure_lineage(failure_lineage)
    _require(
        reuse_admission.payload.get("superseded_empty_lineage")
        in (None, lineage.public_binding)
        and reuse_admission.payload.get("superseded_zero_quality_failure_lineage")
        in (None, failure_lineage.public_binding)
        and preheldout_genesis.payload.get("reuse_admission")
        in (None, reuse_admission.public_binding),
        "Cached v1.3.2 static lineage binding drifted.",
    )
    for binding, label in (
        (reuse_admission.public_binding, "Cached v1.3.2 reuse admission"),
        (preheldout_genesis.public_binding, "Cached v1.3.2 pre-heldout genesis"),
    ):
        _validate_file_binding(
            binding,
            label=label,
            repository_root=quality_context.repository_root,
        )
    if require_live_prestart_absence:
        _v1_3_2_prestart_absent_paths(repository_root=quality_context.repository_root)
    assert_quality_context_unchanged(quality_context)
    return reuse_admission, preheldout_genesis, lineage, failure_lineage


def _revalidate_prestart_quality_authority(
    prestart: PrestartQualityAuthorityV1_3_2,
    *,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> PrestartQualityAuthorityV1_3_2:
    checked = _require_prestart_authority(prestart)
    _revalidate_cached_static_bundle(
        checked,
        quality_context=checked.quality_context,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
        require_live_prestart_absence=True,
    )
    return checked


def _validate_base_prerequisites_binding(
    binding: Mapping[str, Any],
    *,
    prestart: PrestartQualityAuthorityV1_3_2,
    sealed_source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return _validate_base_prerequisites_components(
        binding,
        quality_context=prestart.quality_context,
        reuse_admission=prestart.reuse_admission,
        preheldout_genesis=prestart.preheldout_genesis,
        sealed_source_provenance=sealed_source_provenance,
    )


def _validate_base_prerequisites_components(
    binding: Mapping[str, Any],
    *,
    quality_context: QualityContext,
    reuse_admission: ValidatedReuseAdmission,
    preheldout_genesis: ValidatedPreheldoutGenesis,
    sealed_source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "sealed_source_provenance",
        "manifest",
        "reuse_admission",
        "preheldout_genesis",
        "execution_environment_projection",
        "validated_admitted_calibrations",
        "validated_scale_seed_budget_bundles",
        "quality_execution_topology",
    }
    checked = dict(binding)
    _require(
        set(checked) == expected_fields
        and checked.get("sealed_source_provenance") == dict(sealed_source_provenance)
        and checked.get("manifest") == quality_context.manifest_binding
        and checked.get("reuse_admission") == reuse_admission.public_binding
        and checked.get("preheldout_genesis") == preheldout_genesis.public_binding
        and checked.get("execution_environment_projection")
        == reuse_admission.execution_environment_projection
        and checked.get("validated_admitted_calibrations") == 10
        and checked.get("validated_scale_seed_budget_bundles") == 20
        and isinstance(checked.get("quality_execution_topology"), Mapping)
        and set(cast(Mapping[str, Any], checked["quality_execution_topology"]))
        == {"worker_count", "assignment_rule", "shared_local_filesystem_only"}
        and type(cast(Mapping[str, Any], checked["quality_execution_topology"])["worker_count"])
        is int
        and cast(
            int, cast(Mapping[str, Any], checked["quality_execution_topology"])["worker_count"]
        )
        == 1
        and cast(Mapping[str, Any], checked["quality_execution_topology"])["assignment_rule"]
        == "canonical-coordinate-index-modulo-worker-count-v1"
        and cast(Mapping[str, Any], checked["quality_execution_topology"])[
            "shared_local_filesystem_only"
        ]
        is True,
        "Activation base-prerequisites binding drifted.",
    )
    return checked


def _validate_sealed_source_provenance_v1_3_2(
    value: Mapping[str, Any], *, quality_context: QualityContext
) -> dict[str, Any]:
    checked = dict(value)
    expected_fields = {
        "schema_version",
        "launcher",
        "repository_root",
        "bundle_sha256",
        "pinned_head_oid",
        "frozen_source_commit",
        "implementation_tree_digest",
        "head_manifest",
    }
    head_manifest = checked.get("head_manifest")
    _require(
        set(checked) == expected_fields
        and checked.get("schema_version") == 1
        and checked.get("launcher") == V1_3_2_CANONICAL_GIT_OBJECT_LAUNCHER_ID
        and checked.get("repository_root") == str(quality_context.repository_root)
        and _is_sha256(checked.get("bundle_sha256"))
        and checked.get("pinned_head_oid") == quality_context.source["commit"]
        and checked.get("frozen_source_commit")
        == quality_context.manifest_binding["implementation_source_commit"]
        and checked.get("implementation_tree_digest")
        == quality_context.manifest_binding["implementation_digest"]
        and isinstance(head_manifest, Mapping),
        "Sealed v1.3.2 source provenance drifted.",
    )
    head = cast(Mapping[str, Any], head_manifest)
    _require(
        set(head) == {"path", "git_mode", "git_blob_oid", "sha256", "bytes", "source_base64"}
        and head.get("path") == str(V1_3_2_MANIFEST_RELATIVE_PATH)
        and head.get("git_mode") in {"100644", "100755"}
        and _is_git_oid(head.get("git_blob_oid"))
        and head.get("sha256") == quality_context.manifest_binding["sha256"]
        and head.get("bytes") == quality_context.manifest_binding["bytes"]
        and isinstance(head.get("source_base64"), str),
        "Sealed v1.3.2 HEAD-manifest binding drifted.",
    )
    try:
        source_bytes = base64.b64decode(cast(str, head["source_base64"]), validate=True)
    except (TypeError, ValueError) as error:
        raise ValueError("Sealed v1.3.2 manifest source encoding is invalid.") from error
    opened = _open_secure_regular(
        quality_context.manifest_path,
        label="Live v1.3.2 manifest for sealed provenance",
    )
    try:
        live = opened.read_bytes()
        algorithm = "sha1" if len(cast(str, head["git_blob_oid"])) == 40 else "sha256"
        git_blob = hashlib.new(
            algorithm,
            b"blob " + str(len(live)).encode("ascii") + b"\0" + live,
        ).hexdigest()
        _require(
            source_bytes == live
            and hashlib.sha256(source_bytes).hexdigest() == head["sha256"]
            and git_blob == head["git_blob_oid"],
            "Sealed v1.3.2 manifest bytes differ from the live manifest.",
        )
        opened.assert_unchanged()
    finally:
        opened.close()
    return checked


def _validate_sealed_launch_routing_v1_3_2(
    value: Mapping[str, Any], *, source_provenance: Mapping[str, Any]
) -> dict[str, Any]:
    checked = dict(value)
    _require(
        set(checked)
        == {
            "schema_version",
            "launcher",
            "entrypoint_selector",
            "entrypoint_relative_path",
            "source_bundle_sha256",
            "git_mode",
            "git_blob_oid",
            "sha256",
            "bytes",
        }
        and checked.get("schema_version") == 1
        and checked.get("launcher") == V1_3_2_CANONICAL_GIT_OBJECT_LAUNCHER_ID
        and checked.get("entrypoint_selector") == "matrix"
        and checked.get("entrypoint_relative_path")
        == "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix_v1_3.py"
        and checked.get("source_bundle_sha256") == source_provenance.get("bundle_sha256")
        and checked.get("git_mode") in {"100644", "100755"}
        and _is_git_oid(checked.get("git_blob_oid"))
        and _is_sha256(checked.get("sha256"))
        and type(checked.get("bytes")) is int
        and cast(int, checked["bytes"]) > 0,
        "Sealed v1.3.2 matrix launch routing drifted.",
    )
    return checked


def _activation_policy() -> dict[str, Any]:
    contract = importlib.import_module("p2_direct_controller_contract_v1_3")
    builder = getattr(contract, "expected_v1_3_2_activation_policy", None)
    _require(callable(builder), "V1.3.2 activation policy builder is unavailable.")
    value = cast(Callable[[], object], builder)()
    _require(isinstance(value, Mapping), "V1.3.2 activation policy is invalid.")
    return dict(cast(Mapping[str, Any], value))


def _activation_root_members(
    root: Path,
    *,
    canonical_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _require(os.path.lexists(root), "V1.3.2 activation root is missing.")
    root = _exact_path(root, label="V1.3.2 activation root", must_exist=True)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Activation validation requires O_NOFOLLOW.")
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | cast(int, no_follow),
    )
    try:
        opened_root = os.fstat(descriptor)
        current_root = os.stat(root, follow_symlinks=False)
        _require(
            stat.S_ISDIR(opened_root.st_mode)
            and (opened_root.st_dev, opened_root.st_ino)
            == (current_root.st_dev, current_root.st_ino)
            and opened_root.st_uid == current_root.st_uid == os.getuid()
            and stat.S_IMODE(opened_root.st_mode)
            == stat.S_IMODE(current_root.st_mode)
            == SAFE_DIRECTORY_MODE,
            "Activation root ownership, identity, or mode is unsafe.",
        )
        expected_names = {
            V1_3_2_ACTIVATION_MATRIX_LOCK_PATH.name,
            V1_3_2_QUALITY_START_ACTIVATION_PATH.name,
        }
        _require(set(os.listdir(descriptor)) == expected_names, "Activation root is not exact2.")
        identities: dict[str, os.stat_result] = {}
        for name in sorted(expected_names):
            child_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child_fd)
                current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                _require(
                    stat.S_ISREG(opened.st_mode)
                    and (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
                    and opened.st_uid == current.st_uid == os.getuid()
                    and opened.st_nlink == current.st_nlink == 1
                    and stat.S_IMODE(opened.st_mode)
                    == stat.S_IMODE(current.st_mode)
                    == SAFE_FILE_MODE,
                    f"Activation member metadata is unsafe: {name}",
                )
                identities[name] = opened
            finally:
                os.close(child_fd)
        final_root = os.stat(root, follow_symlinks=False)
        _require(
            (opened_root.st_dev, opened_root.st_ino) == (final_root.st_dev, final_root.st_ino)
            and set(os.listdir(descriptor)) == expected_names,
            "Activation root changed during exact2 validation.",
        )
    finally:
        os.close(descriptor)
    root_binding = {
        "path": str(canonical_root),
        "device": opened_root.st_dev,
        "inode": opened_root.st_ino,
        "uid": opened_root.st_uid,
        "gid": opened_root.st_gid,
        "mode": SAFE_DIRECTORY_MODE,
        "nlink": opened_root.st_nlink,
        "persistent_inode": True,
    }
    lock = identities[V1_3_2_ACTIVATION_MATRIX_LOCK_PATH.name]
    lock_binding = {
        "path": str(canonical_root / V1_3_2_ACTIVATION_MATRIX_LOCK_PATH.name),
        "semantics": V1_3_2_ACTIVATION_MATRIX_LOCK_SEMANTICS,
        "persistent_inode": True,
        "device": lock.st_dev,
        "inode": lock.st_ino,
        "uid": lock.st_uid,
        "mode": SAFE_FILE_MODE,
        "nlink": lock.st_nlink,
        "unlink_on_release": False,
    }
    return (
        root_binding,
        lock_binding,
        root / V1_3_2_QUALITY_START_ACTIVATION_PATH.name,
    )


def _activation_public_binding(
    *, path: Path, payload: Mapping[str, Any], sha256: str, byte_count: int
) -> dict[str, Any]:
    envelope = payload.get("attestation")
    lock = payload.get("matrix_lock_binding")
    source = payload.get("sealed_source_provenance")
    routing = payload.get("sealed_launch_routing")
    failure_lineage = payload.get("superseded_zero_quality_failure_lineage")
    _require(
        all(
            isinstance(value, Mapping)
            for value in (envelope, lock, source, routing, failure_lineage)
        )
        and payload.get("superseded_zero_quality_failure_lineage_projection_sha256")
        == cast(Mapping[str, Any], failure_lineage).get(
            "normalized_contract_projection_sha256"
        ),
        "Activation public-binding inputs are invalid.",
    )
    return {
        "path": str(path),
        "sha256": sha256,
        "bytes": byte_count,
        "experiment_id": payload.get("experiment_id"),
        "payload_sha256": payload.get("payload_sha256"),
        "attestation_mac": cast(Mapping[str, Any], envelope).get("mac"),
        "activation_root": payload.get("canonical_root"),
        "matrix_lock_path": cast(Mapping[str, Any], lock).get("path"),
        "matrix_lock_device": cast(Mapping[str, Any], lock).get("device"),
        "matrix_lock_inode": cast(Mapping[str, Any], lock).get("inode"),
        "base_prerequisites_sha256": payload.get("base_prerequisites_sha256"),
        "sealed_source_bundle_sha256": cast(Mapping[str, Any], source).get("bundle_sha256"),
        "sealed_launch_routing_sha256": _json_digest(dict(cast(Mapping[str, Any], routing))),
        "superseded_failure_lineage_sha256": cast(Mapping[str, Any], failure_lineage).get(
            "lineage_sha256"
        ),
        "superseded_failure_lineage_projection_sha256": payload.get(
            "superseded_zero_quality_failure_lineage_projection_sha256"
        ),
    }


def _build_quality_start_activation_payload(
    *,
    prestart: PrestartQualityAuthorityV1_3_2,
    trust_root: attestation.TrustRoot,
    base_prerequisites_binding: Mapping[str, Any],
    sealed_source_provenance: Mapping[str, Any],
    sealed_launch_routing: Mapping[str, Any],
    root_identity: Mapping[str, Any],
    matrix_lock_binding: Mapping[str, Any],
) -> dict[str, Any]:
    authority = _require_prestart_authority(prestart)
    _require_v1_3_2_context(authority.quality_context, trust_root=trust_root)
    source = _validate_sealed_source_provenance_v1_3_2(
        sealed_source_provenance,
        quality_context=authority.quality_context,
    )
    routing = _validate_sealed_launch_routing_v1_3_2(
        sealed_launch_routing,
        source_provenance=source,
    )
    base = _validate_base_prerequisites_binding(
        base_prerequisites_binding,
        prestart=authority,
        sealed_source_provenance=source,
    )
    genesis = authority.preheldout_genesis.payload
    root = _absolute(
        V1_3_2_ACTIVATION_ROOT,
        repository_root=authority.quality_context.repository_root,
    )
    activation_path = root / V1_3_2_QUALITY_START_ACTIVATION_PATH.name
    return _attested_payload(
        {
            "schema_version": V1_3_2_ACTIVATION_SCHEMA_VERSION,
            "artifact_type": "direct-controller-v1-3-2-quality-start-activation",
            "experiment_id": V1_3_2_QUALITY_EXPERIMENT_ID,
            "status": "activated",
            "canonical_root": str(root),
            "canonical_path": str(activation_path),
            "activation_nonce": secrets.token_hex(32),
            "quality_source": authority.quality_context.source,
            "quality_manifest": authority.quality_context.manifest_binding,
            "reuse_admission": authority.reuse_admission.public_binding,
            "preheldout_genesis": authority.preheldout_genesis.public_binding,
            "superseded_empty_lineage": authority.superseded_empty_lineage.public_binding,
            "superseded_zero_quality_failure_lineage": (
                authority.superseded_failure_lineage.public_binding
            ),
            "superseded_zero_quality_failure_lineage_projection_sha256": (
                _superseded_failure_projection_sha256(
                    authority.superseded_failure_lineage
                )
            ),
            "prestart_absence_witness": authority.absence_witness,
            "base_prerequisites_binding": base,
            "base_prerequisites_sha256": _json_digest(base),
            "sealed_source_provenance": source,
            "sealed_launch_routing": routing,
            "activation_root_identity": dict(root_identity),
            "matrix_lock_binding": dict(matrix_lock_binding),
            "expected_shards": genesis["expected_shards"],
            "coordinate_digest": genesis["coordinate_digest"],
            "exact_fill_arm_names": list(genesis["exact_fill_arm_names"]),
            "activation_policy": _activation_policy(),
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
        purpose=V1_3_2_QUALITY_START_ACTIVATION_PURPOSE,
    )


def _validate_quality_start_activation_internal(
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    storage_root: Path,
    canonical_root: Path,
    sealed_source_provenance: Mapping[str, Any] | None,
    sealed_launch_routing: Mapping[str, Any] | None,
    expected_base_prerequisites_binding: Mapping[str, Any] | None,
    expected_public_binding: Mapping[str, Any] | None,
    consumer_scope: _ActivatedConsumerScopeV1_3_2 | None = None,
    static_authority: (
        PrestartQualityAuthorityV1_3_2 | ValidatedQualityStartActivationV1_3_2 | None
    ) = None,
) -> ValidatedQualityStartActivationV1_3_2:
    _require_v1_3_2_context(quality_context, trust_root=trust_root)
    canonical = _absolute(canonical_root, repository_root=quality_context.repository_root)
    _require(
        canonical
        == _absolute(V1_3_2_ACTIVATION_ROOT, repository_root=quality_context.repository_root),
        "V1.3.2 activation root is not canonical.",
    )
    root_identity, lock_binding, storage_path = _activation_root_members(
        storage_root,
        canonical_root=canonical,
    )
    payload, opened = _load_json_nofollow(
        storage_path,
        label="V1.3.2 quality-start activation",
        require_canonical_pretty_bytes=True,
    )
    try:
        expected_fields = {
            "schema_version",
            "artifact_type",
            "experiment_id",
            "status",
            "canonical_root",
            "canonical_path",
            "activation_nonce",
            "quality_source",
            "quality_manifest",
            "reuse_admission",
            "preheldout_genesis",
            "superseded_empty_lineage",
            "superseded_zero_quality_failure_lineage",
            "superseded_zero_quality_failure_lineage_projection_sha256",
            "prestart_absence_witness",
            "base_prerequisites_binding",
            "base_prerequisites_sha256",
            "sealed_source_provenance",
            "sealed_launch_routing",
            "activation_root_identity",
            "matrix_lock_binding",
            "expected_shards",
            "coordinate_digest",
            "exact_fill_arm_names",
            "activation_policy",
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
            "scientific_subprocesses_started_during_activation",
            "payload_sha256",
            "attestation",
        }
        _require(set(payload) == expected_fields, "Activation receipt schema drifted.")
        _verify_attested_payload(
            payload,
            trust_root=trust_root,
            purpose=V1_3_2_QUALITY_START_ACTIVATION_PURPOSE,
            label="V1.3.2 quality-start activation",
        )
        if static_authority is None:
            admission, genesis, lineage, failure_lineage = _load_v1_3_2_static_bundle(
                quality_context=quality_context,
                trust_root=trust_root,
                expected_shards=expected_shards,
                coordinate_digest=coordinate_digest,
                exact_fill_arm_names=exact_fill_arm_names,
                require_prestart_absence=False,
                consumer_scope=consumer_scope,
            )
        else:
            _require(
                consumer_scope is None,
                "A cached full-grid authority cannot satisfy a scoped consumer replay.",
            )
            admission, genesis, lineage, failure_lineage = _revalidate_cached_static_bundle(
                static_authority,
                quality_context=quality_context,
                trust_root=trust_root,
                expected_shards=expected_shards,
                coordinate_digest=coordinate_digest,
                exact_fill_arm_names=exact_fill_arm_names,
                require_live_prestart_absence=False,
            )
        raw_source = payload.get("sealed_source_provenance")
        raw_routing = payload.get("sealed_launch_routing")
        raw_base = payload.get("base_prerequisites_binding")
        raw_absence_witness = payload.get("prestart_absence_witness")
        _require(
            isinstance(raw_source, Mapping)
            and isinstance(raw_routing, Mapping)
            and isinstance(raw_base, Mapping),
            "Activation sealed source, routing, or prerequisites are missing.",
        )
        _require(
            isinstance(raw_absence_witness, Mapping),
            "Activation prestart absence witness is missing.",
        )
        absence_witness = _validate_prestart_absence_witness(
            cast(Mapping[str, Any], raw_absence_witness),
            quality_context=quality_context,
            reuse_admission=admission,
            preheldout_genesis=genesis,
            superseded_empty_lineage=lineage,
            superseded_failure_lineage=failure_lineage,
        )
        source = _validate_sealed_source_provenance_v1_3_2(
            cast(Mapping[str, Any], raw_source),
            quality_context=quality_context,
        )
        routing = _validate_sealed_launch_routing_v1_3_2(
            cast(Mapping[str, Any], raw_routing),
            source_provenance=source,
        )
        base = _validate_base_prerequisites_components(
            cast(Mapping[str, Any], raw_base),
            quality_context=quality_context,
            reuse_admission=admission,
            preheldout_genesis=genesis,
            sealed_source_provenance=source,
        )
        if sealed_source_provenance is not None:
            _require(
                source == dict(sealed_source_provenance),
                "Activation source differs from active sealed source provenance.",
            )
        if sealed_launch_routing is not None:
            _require(
                routing == dict(sealed_launch_routing),
                "Activation routing differs from active sealed matrix routing.",
            )
        if expected_base_prerequisites_binding is not None:
            _require(
                base == dict(expected_base_prerequisites_binding),
                "Activation base prerequisites differ from the expected binding.",
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
            "scientific_subprocesses_started_during_activation",
        )
        final_path = canonical / V1_3_2_QUALITY_START_ACTIVATION_PATH.name
        _require(
            payload.get("schema_version") == V1_3_2_ACTIVATION_SCHEMA_VERSION
            and payload.get("artifact_type") == "direct-controller-v1-3-2-quality-start-activation"
            and payload.get("experiment_id") == V1_3_2_QUALITY_EXPERIMENT_ID
            and payload.get("status") == "activated"
            and payload.get("canonical_root") == str(canonical)
            and payload.get("canonical_path") == str(final_path)
            and _is_sha256(payload.get("activation_nonce"))
            and payload.get("quality_source") == quality_context.source
            and payload.get("quality_manifest") == quality_context.manifest_binding
            and payload.get("reuse_admission") == admission.public_binding
            and payload.get("preheldout_genesis") == genesis.public_binding
            and payload.get("superseded_empty_lineage") == lineage.public_binding
            and payload.get("superseded_zero_quality_failure_lineage")
            == failure_lineage.public_binding
            and payload.get("superseded_zero_quality_failure_lineage_projection_sha256")
            == _superseded_failure_projection_sha256(failure_lineage)
            and payload.get("prestart_absence_witness") == absence_witness
            and payload.get("base_prerequisites_binding") == base
            and payload.get("base_prerequisites_sha256") == _json_digest(base)
            and payload.get("sealed_source_provenance") == source
            and payload.get("sealed_launch_routing") == routing
            and payload.get("activation_root_identity") == root_identity
            and payload.get("matrix_lock_binding") == lock_binding
            and payload.get("expected_shards") == expected_shards
            and payload.get("coordinate_digest") == coordinate_digest
            and tuple(payload.get("exact_fill_arm_names", ())) == tuple(exact_fill_arm_names)
            and payload.get("activation_policy") == _activation_policy()
            and payload.get("records") == []
            and payload.get("quality_evaluation_started") is False
            and payload.get("evaluation_seed_used_to_initialize_quality_rng") is False
            and payload.get("quality_rng_initialized") is False
            and all(payload.get(field) == 0 for field in zero_fields),
            "Activation receipt identity, zero-prefix, or authority contract drifted.",
        )
        public_binding = _activation_public_binding(
            path=final_path,
            payload=payload,
            sha256=opened.sha256,
            byte_count=opened.bytes,
        )
        if expected_public_binding is not None:
            _require(
                public_binding == dict(expected_public_binding),
                "Activation public binding differs from its authenticated consumer input.",
            )
        opened.assert_unchanged()
    finally:
        opened.close()
    _activation_root_members(storage_root, canonical_root=canonical)
    assert_quality_context_unchanged(quality_context)
    return ValidatedQualityStartActivationV1_3_2(
        _seal=_VALIDATED_QUALITY_START_ACTIVATION_V1_3_2_SEAL,
        payload=dict(payload),
        public_binding=public_binding,
        quality_context=quality_context,
        reuse_admission=admission,
        preheldout_genesis=genesis,
        superseded_empty_lineage=lineage,
        superseded_failure_lineage=failure_lineage,
        consumer_coordinate=(
            None if consumer_scope is None else consumer_scope.coordinate
        ),
        root_identity=root_identity,
        matrix_lock_binding=lock_binding,
    )


def load_activated_quality_authority(
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
    sealed_source_provenance: Mapping[str, Any] | None = None,
    sealed_launch_routing: Mapping[str, Any] | None = None,
    expected_base_prerequisites_binding: Mapping[str, Any] | None = None,
    expected_public_binding: Mapping[str, Any] | None = None,
) -> ValidatedQualityStartActivationV1_3_2:
    """Validate activation read-only; never acquire the matrix flock here."""

    root = _absolute(V1_3_2_ACTIVATION_ROOT, repository_root=quality_context.repository_root)
    return _validate_quality_start_activation_internal(
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
        consumer_scope=None,
    )


def load_activated_consumer_authority(
    *,
    quality_context: QualityContext,
    trust_root: attestation.TrustRoot,
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
) -> ActivatedConsumerAuthorityV1_3_2:
    """Authenticate one admitted coordinate without granting full-owner authority."""

    _require(
        scale in SCALES and training_seed in TRAINING_SEEDS,
        "Activated consumer coordinate is outside the admitted grid.",
    )
    _require(
        isinstance(calibration_binding, Mapping)
        and isinstance(checkpoint_binding, Mapping),
        "Activated consumer artifact bindings must be mappings.",
    )
    coordinate = (scale, training_seed)
    scope = _ActivatedConsumerScopeV1_3_2(
        _seal=_ACTIVATED_CONSUMER_SCOPE_V1_3_2_SEAL,
        coordinate=coordinate,
        calibration_binding=dict(calibration_binding),
        checkpoint_binding=dict(checkpoint_binding),
    )
    root = _absolute(V1_3_2_ACTIVATION_ROOT, repository_root=quality_context.repository_root)
    activation = _validate_quality_start_activation_internal(
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
        consumer_scope=scope,
    )
    _require(
        activation.consumer_coordinate == coordinate
        and set(activation.reuse_admission.calibrations) == {coordinate}
        and set(activation.reuse_admission.checkpoints) == {coordinate},
        "Activated consumer authority escaped its selected coordinate.",
    )
    return require_activated_consumer_authority(
        ActivatedConsumerAuthorityV1_3_2(
            _seal=_ACTIVATED_CONSUMER_AUTHORITY_V1_3_2_SEAL,
            activation=activation,
            reuse_admission=activation.reuse_admission,
            preheldout_genesis=activation.preheldout_genesis,
            coordinate=coordinate,
        )
    )


def _remove_safe_activation_staging(path: Path) -> None:
    metadata = os.stat(path, follow_symlinks=False)
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and stat.S_IMODE(metadata.st_mode) == SAFE_DIRECTORY_MODE,
        f"Unsafe activation staging root requires manual quarantine: {path}",
    )
    allowed = {
        V1_3_2_ACTIVATION_MATRIX_LOCK_PATH.name,
        V1_3_2_QUALITY_START_ACTIVATION_PATH.name,
    }
    with os.scandir(path) as iterator:
        entries = list(iterator)
    for entry in entries:
        child = entry.stat(follow_symlinks=False)
        _require(
            entry.name in allowed
            and stat.S_ISREG(child.st_mode)
            and child.st_uid == os.getuid()
            and child.st_nlink == 1
            and stat.S_IMODE(child.st_mode) == SAFE_FILE_MODE,
            f"Unsafe activation staging content requires quarantine: {entry.path}",
        )
    for entry in entries:
        os.unlink(entry.path)
    _fsync_directory(path, exact_mode=SAFE_DIRECTORY_MODE)
    os.rmdir(path)


def _recover_activation_staging_only(parent: Path) -> None:
    recovered = False
    with os.scandir(parent) as iterator:
        candidates = sorted(
            Path(entry.path)
            for entry in iterator
            if entry.name.startswith(V1_3_2_ACTIVATION_STAGING_PREFIX)
        )
    for candidate in candidates:
        _remove_safe_activation_staging(candidate)
        recovered = True
    if recovered:
        _fsync_directory(parent)


def _create_and_lock_activation_matrix_lock(path: Path) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Activation matrix lock requires O_NOFOLLOW.")
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
        SAFE_FILE_MODE,
    )
    try:
        os.fchmod(descriptor, SAFE_FILE_MODE)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE,
            "New activation matrix lock metadata is unsafe.",
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _pre_receipt_activation_identities(
    staging: Path, *, canonical_root: Path, lock_descriptor: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = os.stat(staging, follow_symlinks=False)
    lock = os.fstat(lock_descriptor)
    current_lock = os.stat(
        staging / V1_3_2_ACTIVATION_MATRIX_LOCK_PATH.name,
        follow_symlinks=False,
    )
    _require(
        stat.S_ISDIR(root.st_mode)
        and root.st_uid == os.getuid()
        and stat.S_IMODE(root.st_mode) == SAFE_DIRECTORY_MODE
        and stat.S_ISREG(lock.st_mode)
        and (lock.st_dev, lock.st_ino) == (current_lock.st_dev, current_lock.st_ino)
        and lock.st_uid == current_lock.st_uid == os.getuid()
        and lock.st_nlink == current_lock.st_nlink == 1
        and stat.S_IMODE(lock.st_mode) == SAFE_FILE_MODE,
        "Staged activation root or lock identity is unsafe.",
    )
    return (
        {
            "path": str(canonical_root),
            "device": root.st_dev,
            "inode": root.st_ino,
            "uid": root.st_uid,
            "gid": root.st_gid,
            "mode": SAFE_DIRECTORY_MODE,
            "nlink": root.st_nlink,
            "persistent_inode": True,
        },
        {
            "path": str(canonical_root / V1_3_2_ACTIVATION_MATRIX_LOCK_PATH.name),
            "semantics": V1_3_2_ACTIVATION_MATRIX_LOCK_SEMANTICS,
            "persistent_inode": True,
            "device": lock.st_dev,
            "inode": lock.st_ino,
            "uid": lock.st_uid,
            "mode": SAFE_FILE_MODE,
            "nlink": lock.st_nlink,
            "unlink_on_release": False,
        },
    )


def publish_quality_start_activation(
    *,
    prestart: PrestartQualityAuthorityV1_3_2,
    trust_root: attestation.TrustRoot,
    base_prerequisites_binding: Mapping[str, Any],
    sealed_source_provenance: Mapping[str, Any],
    sealed_launch_routing: Mapping[str, Any],
) -> QualityStartActivationLeaseV1_3_2:
    """Atomically publish exact2 activation while retaining the same lock FD."""

    authority = _require_prestart_authority(prestart)
    _require_v1_3_2_context(authority.quality_context, trust_root=trust_root)
    expected_shards = authority.preheldout_genesis.payload.get("expected_shards")
    coordinate_digest = authority.preheldout_genesis.payload.get("coordinate_digest")
    exact_fill_arm_names = tuple(
        authority.preheldout_genesis.payload.get("exact_fill_arm_names", ())
    )
    _require(
        type(expected_shards) is int
        and _is_sha256(coordinate_digest)
        and exact_fill_arm_names == FROZEN_EXACT_FILL_ARM_NAMES,
        "Prestart genesis registration is invalid.",
    )
    fresh = _revalidate_prestart_quality_authority(
        authority,
        trust_root=trust_root,
        expected_shards=cast(int, expected_shards),
        coordinate_digest=cast(str, coordinate_digest),
        exact_fill_arm_names=exact_fill_arm_names,
    )
    _require(
        fresh.reuse_admission.public_binding == authority.reuse_admission.public_binding
        and fresh.preheldout_genesis.public_binding == authority.preheldout_genesis.public_binding
        and fresh.superseded_empty_lineage.public_binding
        == authority.superseded_empty_lineage.public_binding
        and fresh.superseded_failure_lineage.public_binding
        == authority.superseded_failure_lineage.public_binding
        and fresh.absence_witness == authority.absence_witness,
        "Prestart authority became stale before activation publication.",
    )
    root = _absolute(
        V1_3_2_ACTIVATION_ROOT,
        repository_root=authority.quality_context.repository_root,
    )
    parent = _exact_path(root.parent, label="Activation parent", must_exist=True)
    from adaptive_v4_gpu_lock import acquire_gpu_lock

    bootstrap = acquire_gpu_lock(
        "p2-direct-controller-v1.3.2-activation-bootstrap",
        path=V1_3_2_ACTIVATION_BOOTSTRAP_LOCK_PATH,
    )
    staging: Path | None = None
    matrix_descriptor: int | None = None
    activation_lease: QualityStartActivationLeaseV1_3_2 | None = None
    published = False
    returned = False
    bootstrap_closed = False
    try:
        bootstrap.assert_held()
        _fsync_directory(parent)
        _require(not os.path.lexists(root), "Final activation root is immutable.")
        _recover_activation_staging_only(parent)
        _require(not os.path.lexists(root), "Activation root appeared during recovery.")
        fresh = _revalidate_prestart_quality_authority(
            authority,
            trust_root=trust_root,
            expected_shards=cast(int, expected_shards),
            coordinate_digest=cast(str, coordinate_digest),
            exact_fill_arm_names=exact_fill_arm_names,
        )
        _require(
            fresh.absence_witness == authority.absence_witness,
            "Prestart absence witness changed under the bootstrap lock.",
        )
        staging = parent / f"{V1_3_2_ACTIVATION_STAGING_PREFIX}{secrets.token_hex(16)}"
        os.mkdir(staging, SAFE_DIRECTORY_MODE)
        os.chmod(staging, SAFE_DIRECTORY_MODE)
        matrix_descriptor = _create_and_lock_activation_matrix_lock(
            staging / V1_3_2_ACTIVATION_MATRIX_LOCK_PATH.name
        )
        root_binding, lock_binding = _pre_receipt_activation_identities(
            staging,
            canonical_root=root,
            lock_descriptor=matrix_descriptor,
        )
        payload = _build_quality_start_activation_payload(
            prestart=fresh,
            trust_root=trust_root,
            base_prerequisites_binding=base_prerequisites_binding,
            sealed_source_provenance=sealed_source_provenance,
            sealed_launch_routing=sealed_launch_routing,
            root_identity=root_binding,
            matrix_lock_binding=lock_binding,
        )
        _write_exclusive_durable(
            staging / V1_3_2_QUALITY_START_ACTIVATION_PATH.name,
            canonical_pretty_json(payload),
        )
        _fsync_directory(staging, exact_mode=SAFE_DIRECTORY_MODE)
        staged = _validate_quality_start_activation_internal(
            quality_context=authority.quality_context,
            trust_root=trust_root,
            expected_shards=cast(int, expected_shards),
            coordinate_digest=cast(str, coordinate_digest),
            exact_fill_arm_names=exact_fill_arm_names,
            storage_root=staging,
            canonical_root=root,
            sealed_source_provenance=sealed_source_provenance,
            sealed_launch_routing=sealed_launch_routing,
            expected_base_prerequisites_binding=base_prerequisites_binding,
            expected_public_binding=None,
            static_authority=fresh,
        )
        bootstrap.assert_held()
        _require(not os.path.lexists(root), "Activation root appeared before no-replace rename.")
        _assert_v1_3_2_quality_paths_absent(
            repository_root=authority.quality_context.repository_root
        )
        _rename_directory_noreplace(staging, root)
        published = True
        _fsync_directory(parent)
        final = _validate_quality_start_activation_internal(
            quality_context=authority.quality_context,
            trust_root=trust_root,
            expected_shards=cast(int, expected_shards),
            coordinate_digest=cast(str, coordinate_digest),
            exact_fill_arm_names=exact_fill_arm_names,
            storage_root=root,
            canonical_root=root,
            sealed_source_provenance=sealed_source_provenance,
            sealed_launch_routing=sealed_launch_routing,
            expected_base_prerequisites_binding=base_prerequisites_binding,
            expected_public_binding=staged.public_binding,
            static_authority=fresh,
        )
        _require(
            final.root_identity == staged.root_identity
            and final.matrix_lock_binding == staged.matrix_lock_binding,
            "Activation root or matrix-lock inode changed across publication.",
        )
        current = os.stat(final.matrix_lock_binding["path"], follow_symlinks=False)
        held = os.fstat(matrix_descriptor)
        _require(
            (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino),
            "Preheld activation matrix-lock FD changed across rename.",
        )
        activation_lease = QualityStartActivationLeaseV1_3_2(
            _QUALITY_START_ACTIVATION_LEASE_V1_3_2_SEAL,
            final,
            matrix_descriptor,
        )
        matrix_descriptor = None
        activation_lease.assert_held()
        bootstrap.close()
        bootstrap_closed = True
        returned = True
        return activation_lease
    finally:
        try:
            try:
                if matrix_descriptor is not None:
                    try:
                        fcntl.flock(matrix_descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(matrix_descriptor)
                if staging is not None and not published and os.path.lexists(staging):
                    _remove_safe_activation_staging(staging)
                    _fsync_directory(parent)
            finally:
                if not bootstrap_closed:
                    bootstrap.close()
        finally:
            if not returned and activation_lease is not None:
                activation_lease.close()
        _require(returned or not published, "Published activation failed final validation.")


def _require_activation_authority(
    activation: ValidatedQualityStartActivationV1_3_2,
) -> ValidatedQualityStartActivationV1_3_2:
    coordinate = getattr(activation, "consumer_coordinate", None)
    expected_coordinates = (
        {(scale, seed) for scale in SCALES for seed in TRAINING_SEEDS}
        if coordinate is None
        else {coordinate}
    )
    _require(
        type(activation) is ValidatedQualityStartActivationV1_3_2
        and activation._seal is _VALIDATED_QUALITY_START_ACTIVATION_V1_3_2_SEAL
        and type(activation.reuse_admission) is ValidatedReuseAdmission
        and type(activation.preheldout_genesis) is ValidatedPreheldoutGenesis
        and type(activation.superseded_empty_lineage) is SupersededEmptyLineageV1_3
        and activation.superseded_empty_lineage._seal is _SUPERSEDED_EMPTY_LINEAGE_SEAL
        and type(activation.superseded_failure_lineage)
        is SupersededZeroQualityFailureLineageV1_3_1
        and activation.superseded_failure_lineage._seal
        is _SUPERSEDED_ZERO_QUALITY_FAILURE_LINEAGE_V1_3_1_SEAL
        and activation.payload.get("reuse_admission")
        == activation.reuse_admission.public_binding
        and activation.payload.get("preheldout_genesis")
        == activation.preheldout_genesis.public_binding
        and activation.payload.get("superseded_empty_lineage")
        == activation.superseded_empty_lineage.public_binding
        and activation.payload.get("superseded_zero_quality_failure_lineage")
        == activation.superseded_failure_lineage.public_binding
        and (
            coordinate is None
            or (
                type(coordinate) is tuple
                and len(coordinate) == 2
                and coordinate[0] in SCALES
                and coordinate[1] in TRAINING_SEEDS
            )
        )
        and set(activation.reuse_admission.calibrations) == expected_coordinates
        and set(activation.reuse_admission.checkpoints) == expected_coordinates,
        "Activated quality authority is raw or duck-typed.",
    )
    return activation


def _require_full_activation_authority(
    activation: ValidatedQualityStartActivationV1_3_2,
) -> ValidatedQualityStartActivationV1_3_2:
    authority = _require_activation_authority(activation)
    _require(
        authority.consumer_coordinate is None,
        "A coordinate-scoped consumer authority cannot mutate or claim the full activation.",
    )
    return authority


def require_activated_consumer_authority(
    consumer: ActivatedConsumerAuthorityV1_3_2,
) -> ActivatedConsumerAuthorityV1_3_2:
    """Reject nominal or duck-typed objects at a cached consumer boundary."""

    _require(
        type(consumer) is ActivatedConsumerAuthorityV1_3_2
        and consumer._seal is _ACTIVATED_CONSUMER_AUTHORITY_V1_3_2_SEAL,
        "Activated consumer authority is raw or duck-typed.",
    )
    activation = _require_activation_authority(consumer.activation)
    _require(
        activation.consumer_coordinate == consumer.coordinate
        and consumer.reuse_admission is activation.reuse_admission
        and consumer.preheldout_genesis is activation.preheldout_genesis
        and set(consumer.reuse_admission.calibrations) == {consumer.coordinate}
        and set(consumer.reuse_admission.checkpoints) == {consumer.coordinate},
        "Activated consumer authority seal or coordinate binding drifted.",
    )
    return consumer


def _reload_activation_capability(
    activation: ValidatedQualityStartActivationV1_3_2,
    *,
    trust_root: attestation.TrustRoot,
) -> ValidatedQualityStartActivationV1_3_2:
    authority = _require_full_activation_authority(activation)
    payload = authority.payload
    root = _absolute(
        V1_3_2_ACTIVATION_ROOT,
        repository_root=authority.quality_context.repository_root,
    )
    return _validate_quality_start_activation_internal(
        quality_context=authority.quality_context,
        trust_root=trust_root,
        expected_shards=cast(int, payload["expected_shards"]),
        coordinate_digest=cast(str, payload["coordinate_digest"]),
        exact_fill_arm_names=cast(Sequence[str], payload["exact_fill_arm_names"]),
        storage_root=root,
        canonical_root=root,
        sealed_source_provenance=cast(Mapping[str, Any], payload["sealed_source_provenance"]),
        sealed_launch_routing=cast(Mapping[str, Any], payload["sealed_launch_routing"]),
        expected_base_prerequisites_binding=cast(
            Mapping[str, Any], payload["base_prerequisites_binding"]
        ),
        expected_public_binding=authority.public_binding,
        static_authority=authority,
    )


def revalidate_activated_quality_authority(
    activation: ValidatedQualityStartActivationV1_3_2,
    *,
    trust_root: attestation.TrustRoot,
) -> ValidatedQualityStartActivationV1_3_2:
    """Recheck an existing full-grid capability without another checkpoint-grid replay."""

    return _reload_activation_capability(activation, trust_root=trust_root)


def acquire_quality_start_activation_lease(
    activation: ValidatedQualityStartActivationV1_3_2,
    *,
    trust_root: attestation.TrustRoot,
) -> QualityStartActivationLeaseV1_3_2:
    """Acquire an existing persistent lock without O_CREAT, then revalidate."""

    authority = _reload_activation_capability(activation, trust_root=trust_root)
    path = Path(cast(str, authority.matrix_lock_binding["path"]))
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(no_follow is not None, "Activation lease acquisition requires O_NOFOLLOW.")
    descriptor = os.open(
        path,
        os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | cast(int, no_follow),
    )
    acquired = False
    reserved_identity: tuple[int, int] | None = None
    lease: QualityStartActivationLeaseV1_3_2 | None = None
    returned = False
    try:
        metadata = os.fstat(descriptor)
        binding = authority.matrix_lock_binding
        _require(
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_dev == binding["device"]
            and metadata.st_ino == binding["inode"]
            and metadata.st_uid == os.getuid()
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == SAFE_FILE_MODE,
            "Existing activation matrix lock differs from its receipt binding.",
        )
        with _ACTIVE_V1_3_2_ACTIVATION_LEASES_GUARD:
            identity = (metadata.st_dev, metadata.st_ino)
            for active_fd in _ACTIVE_V1_3_2_ACTIVATION_LEASE_FDS:
                active = os.fstat(active_fd)
                _require(
                    (active.st_dev, active.st_ino) != identity,
                    "Activation matrix-lock reentry in one process is forbidden.",
                )
            _require(
                identity not in _PENDING_V1_3_2_ACTIVATION_LEASE_IDENTITIES,
                "Activation matrix-lock acquisition is already pending in this process.",
            )
            _PENDING_V1_3_2_ACTIVATION_LEASE_IDENTITIES.add(identity)
            reserved_identity = identity
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = True
        revalidated = _reload_activation_capability(authority, trust_root=trust_root)
        activation_root = _exact_path(
            Path(cast(str, revalidated.root_identity["path"])),
            label="Activation durability recovery root",
            repository_root=revalidated.quality_context.repository_root,
            must_exist=True,
        )
        _fsync_directory(activation_root, exact_mode=SAFE_DIRECTORY_MODE)
        _fsync_directory(activation_root.parent)
        lease = QualityStartActivationLeaseV1_3_2(
            _QUALITY_START_ACTIVATION_LEASE_V1_3_2_SEAL,
            revalidated,
            descriptor,
        )
        descriptor = -1
        lease.assert_held()
        returned = True
        return lease
    finally:
        try:
            if descriptor >= 0:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            elif not returned and lease is not None:
                lease.close()
        finally:
            if reserved_identity is not None:
                with _ACTIVE_V1_3_2_ACTIVATION_LEASES_GUARD:
                    _PENDING_V1_3_2_ACTIVATION_LEASE_IDENTITIES.discard(
                        reserved_identity
                    )


def load_activated_static_bundle(
    activation: ValidatedQualityStartActivationV1_3_2,
    *,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> tuple[ValidatedReuseAdmission, ValidatedPreheldoutGenesis]:
    """Reuse one fully validated activation replay without recursively replaying it."""

    authority = _require_full_activation_authority(activation)
    _require_v1_3_2_context(authority.quality_context, trust_root=trust_root)
    _require(
        expected_shards == authority.payload.get("expected_shards")
        and coordinate_digest == authority.payload.get("coordinate_digest")
        and tuple(exact_fill_arm_names)
        == tuple(authority.payload.get("exact_fill_arm_names", ())),
        "Activated static-bundle registration differs from the receipt.",
    )
    _verify_attested_payload(
        authority.payload,
        trust_root=trust_root,
        purpose=V1_3_2_QUALITY_START_ACTIVATION_PURPOSE,
        label="Cached v1.3.2 quality-start activation",
    )
    _verify_attested_payload(
        authority.reuse_admission.payload,
        trust_root=trust_root,
        purpose=V1_3_2_REUSE_ADMISSION_PURPOSE,
        label="Cached v1.3.2 reuse admission",
    )
    _verify_attested_payload(
        authority.preheldout_genesis.payload,
        trust_root=trust_root,
        purpose=V1_3_2_PREHELDOUT_GENESIS_PURPOSE,
        label="Cached v1.3.2 pre-heldout genesis",
    )
    _require_superseded_lineage(authority.superseded_empty_lineage)
    _require_superseded_failure_lineage(authority.superseded_failure_lineage)
    for binding, label in (
        (authority.public_binding, "Cached v1.3.2 activation"),
        (authority.reuse_admission.public_binding, "Cached v1.3.2 reuse admission"),
        (authority.preheldout_genesis.public_binding, "Cached v1.3.2 pre-heldout genesis"),
    ):
        _require(isinstance(binding, Mapping), f"{label} binding is missing.")
        _validate_file_binding(
            cast(Mapping[str, Any], binding),
            label=label,
            repository_root=authority.quality_context.repository_root,
        )
    assert_quality_context_unchanged(authority.quality_context)
    return authority.reuse_admission, authority.preheldout_genesis


def load_activated_reuse_admission(
    activation: ValidatedQualityStartActivationV1_3_2,
    *,
    trust_root: attestation.TrustRoot,
) -> ValidatedReuseAdmission:
    authority = _require_full_activation_authority(activation)
    validated_admission, _validated_genesis = load_activated_static_bundle(
        authority,
        trust_root=trust_root,
        expected_shards=cast(int, authority.payload["expected_shards"]),
        coordinate_digest=cast(str, authority.payload["coordinate_digest"]),
        exact_fill_arm_names=cast(Sequence[str], authority.payload["exact_fill_arm_names"]),
    )
    return validated_admission


def load_activated_preheldout_genesis(
    activation: ValidatedQualityStartActivationV1_3_2,
    *,
    trust_root: attestation.TrustRoot,
    expected_shards: int,
    coordinate_digest: str,
    exact_fill_arm_names: Sequence[str],
) -> ValidatedPreheldoutGenesis:
    authority = _require_full_activation_authority(activation)
    _validated_admission, validated_genesis = load_activated_static_bundle(
        authority,
        trust_root=trust_root,
        expected_shards=expected_shards,
        coordinate_digest=coordinate_digest,
        exact_fill_arm_names=exact_fill_arm_names,
    )
    return validated_genesis


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
