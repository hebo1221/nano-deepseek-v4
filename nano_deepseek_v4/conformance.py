"""Run fixed, evidence-carrying nano-deepseek-v4 conformance profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import socket
import sys
import tempfile
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata
from importlib.resources import as_file
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from . import __version__
from .architecture import inspect_architecture
from .attention_reach import run_attention_reach
from .config import DeepSeekV4Config
from .demo import run_demo
from .dspark import run_dspark_conformance as run_dspark_vector_conformance
from .dspark_scheduler import run_dspark_scheduler_conformance
from .official_parity import run_transformers_parity
from .verify_flash_0731_receipt import (
    ReceiptUnavailableError,
    ReceiptVerificationError,
    verify_flash_0731_receipt,
)

CheckStatus = Literal["pass", "fail", "skip", "error"]
SuiteStatus = Literal["pass", "fail", "incomplete", "error"]

_SCHEMA_VERSION = 1
_PARITY_VERSION = "5.15.0"
_MIN_HUGGINGFACE_HUB_VERSION = "0.34.0"
_DSPARK_CHECK_ID = "dspark_vectors"
_DSPARK_SCHEDULER_CHECK_ID = "dspark_scheduler"
_CHECK_ORDER = (
    "tiny_model_cache",
    "flash_0731_architecture",
    "attention_reach",
    "transformers_parity",
    "flash_0731_receipt",
)
_PROFILE_CHECKS = {
    "core": _CHECK_ORDER[:3],
    "dspark": (*_CHECK_ORDER[:3], _DSPARK_CHECK_ID, _DSPARK_SCHEDULER_CHECK_ID),
    "offline": _CHECK_ORDER[:4],
    "live": _CHECK_ORDER,
}
_CHECK_MODES = {
    "tiny_model_cache": "offline_local",
    "flash_0731_architecture": "offline_local",
    "attention_reach": "offline_local",
    _DSPARK_CHECK_ID: "offline_local",
    _DSPARK_SCHEDULER_CHECK_ID: "offline_local",
    "transformers_parity": "offline_reference",
    "flash_0731_receipt": "live_network",
}
_CHECK_BOUNDARIES = {
    "tiny_model_cache": (
        "A no-download random-tiny native model full-forward/cache check; not an "
        "official-checkpoint, training-quality, or accelerator-performance claim."
    ),
    "flash_0731_architecture": (
        "An allocation-free consistency check between the packaged Flash-0731 preset "
        "and bundled pinned config; not official weight loading or DSpark execution."
    ),
    "attention_reach": (
        "A fixed-seed random-tiny perturbation and local-gradient reachability check; "
        "not learned retrieval, quality, efficiency, or optimized-kernel evidence."
    ),
    _DSPARK_CHECK_ID: (
        "Tiny eager FP32 CPU agreement among a separate native path, independent "
        "dense oracle, and fixed packaged DSpark vectors; not official-checkpoint, "
        "scheduler, acceptance, kernel, performance, serving, training, or quality evidence."
    ),
    _DSPARK_SCHEDULER_CHECK_ID: (
        "Fixed CPU agreement among the DSpark Algorithm 1 and Section 5.2 scheduler, "
        "an exhaustive oracle, and packaged probability/SPS vectors; not confidence "
        "calibration, target acceptance, hardware profiling, kernels, serving speed, "
        "or end-to-end losslessness evidence."
    ),
    "transformers_parity": (
        "Eager floating-point differential parity on a deterministic tiny fixture "
        "against pinned Transformers source and official MTP equations; not quantized, "
        "distributed, frontier-scale, or optimized-kernel parity."
    ),
    "flash_0731_receipt": (
        "A pinned metadata and selected-shard-header replay; not weight payload "
        "integrity, full-snapshot presence, model quality, or native DSpark execution."
    ),
}
_SUITE_BOUNDARY = (
    "This suite reports selected random-tiny structural and cache checks, fixed-seed "
    "attention reachability, eager floating-point differential parity against pinned "
    "source when requested, and a metadata-only pinned Hub replay when requested. It "
    "does not establish official-weight runtime parity, optimized-kernel correctness, "
    "distributed or long-context behavior, performance, training quality, or model quality."
)
_DSPARK_SUITE_BOUNDARY = (
    "This profile combines the core random-tiny package checks with fixed, offline "
    "DSpark draft-equation and prefix-scheduler vectors. It establishes only the tiny "
    "eager FP32 draft equations plus Algorithm 1 and Section 5.2 scheduler arithmetic "
    "on synthetic probability/SPS tables; it does not establish official-checkpoint "
    "execution, confidence calibration, speculative acceptance, end-to-end losslessness, "
    "optimized kernels, performance, serving capacity, training equivalence, or quality."
)
_REASON_MESSAGES = {
    "not_selected_by_profile": "not selected by this profile",
    "missing_optional_dependency": "required optional dependency is not installed",
    "incompatible_optional_dependency": "installed optional dependency is incompatible",
    "contract_failed": "the check ran and its contract did not hold",
    "receipt_drift": "the pinned receipt or observed metadata drifted",
    "source_unavailable": "the pinned source could not be fetched",
    "inspection_unavailable": "the selected-header inspection could not complete",
    "network_policy_violation": "an offline check attempted forbidden IP network access",
    "unexpected_exception": "the conformance harness encountered an unexpected exception",
}
_EXIT_CODES: dict[SuiteStatus, int] = {
    "pass": 0,
    "fail": 1,
    "incomplete": 3,
    "error": 4,
}
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
_TRUE_ENVIRONMENT_VALUES = frozenset({"1", "ON", "YES", "TRUE"})
_OS_ISOLATION_ENV = "NANO_DEEPSEEK_V4_OS_NETWORK_ISOLATION"
_PARENT_NETNS_ENV = "NANO_DEEPSEEK_V4_PARENT_NETNS"
_EXPECTED_UID_ENV = "NANO_DEEPSEEK_V4_EXPECTED_UID"
_EXPECTED_GID_ENV = "NANO_DEEPSEEK_V4_EXPECTED_GID"
_LINUX_NETNS = "linux_network_namespace"
_PYTHON_GUARD_SCOPE = "best_effort_python_stdlib_calls"
_LINUX_NETNS_SCOPE = "linux_network_namespace_process_tree"
_NETWORK_LOCK = threading.RLock()
_NETWORK_AUDIT_INSTALLED = False
_ACTIVE_NETWORK_GUARD: _NetworkGuardState | None = None


class _NetworkPolicyViolation(RuntimeError):
    """An IP-network operation blocked by the offline conformance guard."""


@dataclass
class _NetworkGuardState:
    blocked_attempt_count: int = 0

    def block(self) -> None:
        self.blocked_attempt_count += 1
        raise _NetworkPolicyViolation("offline conformance forbids IP network access")


def _is_ip_family(value: object) -> bool:
    return value in {socket.AF_INET, socket.AF_INET6}


def _network_audit_hook(event: str, args: tuple[Any, ...]) -> None:
    state = _ACTIVE_NETWORK_GUARD
    if state is None:
        return
    if event == "socket.__new__" and len(args) > 1 and _is_ip_family(args[1]):
        state.block()
    if event in {
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.gethostbyname_ex",
        "socket.getnameinfo",
    }:
        state.block()
    if event in {"socket.bind", "socket.connect", "socket.sendto"} and args:
        family = getattr(args[0], "family", None)
        if _is_ip_family(family):
            state.block()


def _ensure_network_audit_hook() -> None:
    global _NETWORK_AUDIT_INSTALLED
    if not _NETWORK_AUDIT_INSTALLED:
        sys.addaudithook(_network_audit_hook)
        _NETWORK_AUDIT_INSTALLED = True


def _blocked_socket_method(
    original: Any,
    state: _NetworkGuardState,
) -> Any:
    def guarded(sock: socket.socket, *args: Any, **kwargs: Any) -> Any:
        if _is_ip_family(getattr(sock, "family", None)):
            state.block()
        return original(sock, *args, **kwargs)

    return guarded


def _environment_flag(*names: str) -> bool:
    """Parse the first non-empty variable, matching Hub's ``or`` precedence."""

    for name in names:
        value = os.environ.get(name)
        if value:
            return value.upper() in _TRUE_ENVIRONMENT_VALUES
    return False


def _environment_any_flag(*names: str) -> bool:
    """Return whether any variable is true, matching Hub telemetry semantics."""

    return any(
        value is not None and value.upper() in _TRUE_ENVIRONMENT_VALUES
        for name in names
        if (value := os.environ.get(name)) is not None
    )


def _synchronize_cached_huggingface_environment() -> None:
    """Keep lazily imported Hub state aligned with the current environment.

    Hugging Face reads its offline and telemetry flags when
    ``huggingface_hub.constants`` is imported. The offline parity check may be
    the first import in the process, while this module temporarily sets the
    offline environment. Without resynchronizing the cached booleans, the
    cumulative live profile remains offline after the environment is restored.

    Hub 0.x can also cache a ``requests.Session`` whose adapters were selected
    from the cached offline flag. Hub 1.x keeps a global ``httpx.Client``.
    Discard either cache after changing the booleans so 0.x selects adapters
    again and no Hub client spans the policy transition. Only already-imported
    modules are touched; conformance does not import Hub solely to mutate its
    process state.
    """

    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        expected = {
            "HF_HUB_OFFLINE": _environment_flag(
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            ),
            "HF_HUB_DISABLE_TELEMETRY": _environment_any_flag(
                "HF_HUB_DISABLE_TELEMETRY",
                "DISABLE_TELEMETRY",
                "DO_NOT_TRACK",
            ),
        }
        for name, value in expected.items():
            if isinstance(getattr(constants, name, None), bool):
                setattr(constants, name, value)

    http = sys.modules.get("huggingface_hub.utils._http")
    if http is None:
        return
    for method_name in ("reset_sessions", "close_session"):
        reset = getattr(http, method_name, None)
        if callable(reset):
            reset()
            break


@contextmanager
def _isolated_parity_process_state() -> Iterator[None]:
    """Restore caller RNG state and contain lazy TorchInductor cache setup."""

    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    previous_inductor_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    with tempfile.TemporaryDirectory(prefix="nano-dsv4-parity-cache-") as cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        try:
            with torch.random.fork_rng(devices=[]):
                yield
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            if previous_inductor_cache is None:
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
            else:
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_inductor_cache


@contextmanager
def _offline_network_guard() -> Iterator[_NetworkGuardState]:
    global _ACTIVE_NETWORK_GUARD
    _ensure_network_audit_hook()
    if _ACTIVE_NETWORK_GUARD is not None:
        raise RuntimeError("offline network guard is already active")
    state = _NetworkGuardState()
    originals: dict[str, Any] = {}
    method_names = (
        "accept",
        "bind",
        "connect",
        "connect_ex",
        "listen",
        "recv",
        "recv_into",
        "recvfrom",
        "recvfrom_into",
        "recvmsg",
        "recvmsg_into",
        "send",
        "sendall",
        "sendmsg",
        "sendto",
    )
    previous_environment = {
        name: os.environ.get(name) for name in _OFFLINE_ENVIRONMENT
    }
    _ACTIVE_NETWORK_GUARD = state
    try:
        for name in method_names:
            original = getattr(socket.socket, name, None)
            if original is not None:
                originals[name] = original
                setattr(socket.socket, name, _blocked_socket_method(original, state))
        os.environ.update(_OFFLINE_ENVIRONMENT)
        _synchronize_cached_huggingface_environment()
        try:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except _NetworkPolicyViolation:
            state.blocked_attempt_count = 0
        else:  # pragma: no cover - fail-closed platform guard
            raise RuntimeError("Python socket audit hook did not block AF_INET")
        yield state
    finally:
        try:
            for name, original in originals.items():
                setattr(socket.socket, name, original)
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            _synchronize_cached_huggingface_environment()
        finally:
            _ACTIVE_NETWORK_GUARD = None


def _status_integer(status_text: str, name: str, *, base: int) -> int | None:
    for line in status_text.splitlines():
        if line.startswith(f"{name}:"):
            try:
                return int(line.split(":", 1)[1].strip(), base)
            except ValueError:
                return None
    return None


def _process_isolation_verified(
    status_text: str,
    *,
    expected_uid: int,
    expected_gid: int,
    current_uid: int,
    current_gid: int,
    supplementary_groups: set[int],
) -> bool:
    if expected_uid == 0 or expected_gid == 0:
        return False
    if current_uid != expected_uid or current_gid != expected_gid:
        return False
    if supplementary_groups:
        return False
    if any(
        _status_integer(status_text, name, base=16) != 0
        for name in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
    ):
        return False
    return _status_integer(status_text, "NoNewPrivs", base=10) == 1


def _os_network_isolation() -> str:
    if os.environ.get(_OS_ISOLATION_ENV) != _LINUX_NETNS:
        return "none"
    parent_netns = os.environ.get(_PARENT_NETNS_ENV)
    try:
        expected_uid = int(os.environ.get(_EXPECTED_UID_ENV, ""))
        expected_gid = int(os.environ.get(_EXPECTED_GID_ENV, ""))
    except ValueError:
        return "none"
    if not parent_netns or platform.system() != "Linux":
        return "none"
    try:
        current_netns = os.readlink("/proc/self/ns/net")
        interfaces = {name for _, name in socket.if_nameindex()}
        status_text = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return "none"
    if (
        current_netns == parent_netns
        or interfaces != {"lo"}
        or not _process_isolation_verified(
            status_text,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            current_uid=os.geteuid(),
            current_gid=os.getegid(),
            supplementary_groups=set(os.getgroups()),
        )
    ):
        return "none"
    return _LINUX_NETNS


@dataclass(frozen=True)
class ConformanceCheck:
    """One fixed check and its canonical nested receipt."""

    id: str
    required: bool
    mode: str
    status: CheckStatus
    reason_code: str | None
    claim_boundary: str
    receipt_schema_version: int | None
    receipt_sha256: str | None
    receipt: dict[str, Any] | None


@dataclass(frozen=True)
class ConformanceReport:
    """Deterministic aggregate report for one conformance profile."""

    schema_version: int
    kind: str
    profile: str
    status: SuiteStatus
    passed: bool
    complete: bool
    package: dict[str, str]
    environment: dict[str, str | None]
    network: dict[str, str | bool | int]
    implementation_sha256: str
    checks: tuple[ConformanceCheck, ...]
    summary: dict[str, int]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _receipt_sha256(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _finished_check(
    check_id: str,
    *,
    required: bool,
    receipt: dict[str, Any],
    passed: bool,
    reason_code: str = "contract_failed",
    claim_boundary: str | None = None,
) -> ConformanceCheck:
    schema_version = receipt.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        schema_version = None
    return ConformanceCheck(
        id=check_id,
        required=required,
        mode=_CHECK_MODES[check_id],
        status="pass" if passed else "fail",
        reason_code=None if passed else reason_code,
        claim_boundary=claim_boundary or _CHECK_BOUNDARIES[check_id],
        receipt_schema_version=schema_version,
        receipt_sha256=_receipt_sha256(receipt),
        receipt=receipt,
    )


def _skip_check(
    check_id: str,
    *,
    required: bool,
    reason_code: str,
) -> ConformanceCheck:
    return ConformanceCheck(
        id=check_id,
        required=required,
        mode=_CHECK_MODES[check_id],
        status="skip",
        reason_code=reason_code,
        claim_boundary=_CHECK_BOUNDARIES[check_id],
        receipt_schema_version=None,
        receipt_sha256=None,
        receipt=None,
    )


def _error_check(
    check_id: str,
    *,
    required: bool,
    reason_code: str = "unexpected_exception",
) -> ConformanceCheck:
    return ConformanceCheck(
        id=check_id,
        required=required,
        mode=_CHECK_MODES[check_id],
        status="error",
        reason_code=reason_code,
        claim_boundary=_CHECK_BOUNDARIES[check_id],
        receipt_schema_version=None,
        receipt_sha256=None,
        receipt=None,
    )


def _run_demo_check() -> ConformanceCheck:
    check_id = "tiny_model_cache"
    try:
        report = run_demo()
        return _finished_check(
            check_id,
            required=True,
            receipt=report.to_dict(),
            passed=report.passed,
            claim_boundary=report.claim_boundary,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _error_check(check_id, required=True)


def _architecture_receipt() -> tuple[dict[str, Any], bool]:
    preset = DeepSeekV4Config.flash_0731()
    resource = resource_files("nano_deepseek_v4").joinpath(
        "_receipts/DeepSeek-V4-Flash-0731-config.json"
    )
    bundled_bytes = resource.read_bytes()
    with as_file(resource) as path:
        bundled = DeepSeekV4Config.from_official_json(path)
    report = inspect_architecture(preset, source="preset:flash-0731")
    assertions = {
        "normalized_config_matches_bundled_receipt": preset.to_dict() == bundled.to_dict(),
        "auxiliary_kind_is_dspark": report.auxiliary_kind == "dspark",
        "dspark_stage_count_is_three": report.dspark_stage_count == 3,
        "dspark_target_layers_are_terminal": report.dspark_target_layer_ids == [40, 41, 42],
        "runtime_load_is_explicitly_unsupported": report.runtime_load_supported is False,
    }
    receipt = {
        "schema_version": 1,
        "kind": "packaged-flash-0731-architecture-consistency",
        "bundled_config_sha256": hashlib.sha256(bundled_bytes).hexdigest(),
        "architecture": report.to_dict(),
        "assertions": assertions,
        "claim_boundary": _CHECK_BOUNDARIES["flash_0731_architecture"],
    }
    return receipt, all(assertions.values())


def _run_architecture_check() -> ConformanceCheck:
    check_id = "flash_0731_architecture"
    try:
        receipt, passed = _architecture_receipt()
        return _finished_check(
            check_id,
            required=True,
            receipt=receipt,
            passed=passed,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _error_check(check_id, required=True)


def _run_attention_check() -> ConformanceCheck:
    check_id = "attention_reach"
    try:
        report = run_attention_reach(device="cpu")
        return _finished_check(
            check_id,
            required=True,
            receipt=report.to_dict(),
            passed=report.passed,
            claim_boundary=report.claim_boundary,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _error_check(check_id, required=True)


def _run_dspark_check() -> ConformanceCheck:
    check_id = _DSPARK_CHECK_ID
    try:
        report = run_dspark_vector_conformance()
        return _finished_check(
            check_id,
            required=True,
            receipt=report.to_dict(),
            passed=report.passed,
            claim_boundary=report.claim_boundary,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _error_check(check_id, required=True)


def _run_dspark_scheduler_check() -> ConformanceCheck:
    check_id = _DSPARK_SCHEDULER_CHECK_ID
    try:
        report = run_dspark_scheduler_conformance()
        return _finished_check(
            check_id,
            required=True,
            receipt=report.to_dict(),
            passed=report.passed,
            claim_boundary=report.claim_boundary,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _error_check(check_id, required=True)


def _run_parity_check(transformers_version: str | None) -> ConformanceCheck:
    check_id = "transformers_parity"
    if transformers_version is None:
        return _skip_check(
            check_id,
            required=True,
            reason_code="missing_optional_dependency",
        )
    if transformers_version != _PARITY_VERSION:
        return _skip_check(
            check_id,
            required=True,
            reason_code="incompatible_optional_dependency",
        )
    try:
        with _isolated_parity_process_state():
            report = run_transformers_parity()
        return _finished_check(
            check_id,
            required=True,
            receipt=report.to_dict(),
            passed=report.passed,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        return _error_check(check_id, required=True)


def _run_live_check(
    huggingface_hub_version: str | None,
) -> tuple[ConformanceCheck, bool]:
    check_id = "flash_0731_receipt"
    if huggingface_hub_version is None:
        return (
            _skip_check(
                check_id,
                required=True,
                reason_code="missing_optional_dependency",
            ),
            False,
        )
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return (
            _skip_check(
                check_id,
                required=True,
                reason_code="missing_optional_dependency",
            ),
            False,
        )
    try:
        observed_release = Version(huggingface_hub_version)
    except InvalidVersion:
        observed_release = None
    if observed_release is None or observed_release < Version(_MIN_HUGGINGFACE_HUB_VERSION):
        return (
            _skip_check(
                check_id,
                required=True,
                reason_code="incompatible_optional_dependency",
            ),
            False,
        )
    network_attempted = False

    def mark_network_attempt() -> None:
        nonlocal network_attempted
        network_attempted = True

    try:
        with tempfile.TemporaryDirectory(prefix="nano-dsv4-conformance-") as temp:
            receipt = verify_flash_0731_receipt(
                cache_dir=Path(temp),
                on_network_attempt=mark_network_attempt,
            )
        return (
            _finished_check(
                check_id,
                required=True,
                receipt=receipt,
                passed=receipt.get("status") == "pass",
                reason_code="receipt_drift",
                claim_boundary=str(receipt.get("scope_boundary", _CHECK_BOUNDARIES[check_id])),
            ),
            network_attempted,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except ReceiptUnavailableError as exc:
        return (
            _skip_check(
                check_id,
                required=True,
                reason_code=exc.reason_code,
            ),
            network_attempted,
        )
    except ReceiptVerificationError:
        return (
            ConformanceCheck(
                id=check_id,
                required=True,
                mode=_CHECK_MODES[check_id],
                status="fail",
                reason_code="receipt_drift",
                claim_boundary=_CHECK_BOUNDARIES[check_id],
                receipt_schema_version=None,
                receipt_sha256=None,
                receipt=None,
            ),
            network_attempted,
        )
    except Exception:
        return _error_check(check_id, required=True), network_attempted


def _suite_status(checks: Sequence[ConformanceCheck]) -> SuiteStatus:
    required = [check for check in checks if check.required]
    if any(check.status == "error" for check in required):
        return "error"
    if any(check.status == "fail" for check in required):
        return "fail"
    if any(check.status == "skip" for check in required):
        return "incomplete"
    return "pass"


def _run_dspark_profile_locked(
    *,
    os_isolation: str,
) -> ConformanceReport:
    checks: list[ConformanceCheck] = []
    accounted_attempts = 0
    selected_order = (
        *_CHECK_ORDER[:3],
        _DSPARK_CHECK_ID,
        _DSPARK_SCHEDULER_CHECK_ID,
    )
    with _offline_network_guard() as guard:
        for check_id in selected_order:
            if check_id == "tiny_model_cache":
                check = _run_demo_check()
            elif check_id == "flash_0731_architecture":
                check = _run_architecture_check()
            elif check_id == "attention_reach":
                check = _run_attention_check()
            elif check_id == _DSPARK_CHECK_ID:
                check = _run_dspark_check()
            else:
                check = _run_dspark_scheduler_check()
            checks.append(check)
            if guard.blocked_attempt_count > accounted_attempts:
                checks[-1] = _error_check(
                    check_id,
                    required=True,
                    reason_code="network_policy_violation",
                )
                accounted_attempts = guard.blocked_attempt_count

    blocked_attempt_count = guard.blocked_attempt_count
    if blocked_attempt_count > accounted_attempts:
        checks[-1] = _error_check(
            checks[-1].id,
            required=True,
            reason_code="network_policy_violation",
        )
    checks.extend(
        (
            _skip_check(
                "transformers_parity",
                required=False,
                reason_code="not_selected_by_profile",
            ),
            _skip_check(
                "flash_0731_receipt",
                required=False,
                reason_code="not_selected_by_profile",
            ),
        )
    )
    status = _suite_status(checks)
    summary = {
        state: sum(check.status == state for check in checks)
        for state in ("pass", "fail", "skip", "error")
    }
    summary["required"] = sum(check.required for check in checks)
    complete = all(
        not check.required or check.status in {"pass", "fail"}
        for check in checks
    )
    return ConformanceReport(
        schema_version=_SCHEMA_VERSION,
        kind="nano-deepseek-v4-conformance",
        profile="dspark",
        status=status,
        passed=status == "pass",
        complete=complete,
        package={"name": "nano-deepseek-v4", "version": __version__},
        environment={
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "platform": platform.system(),
            "machine": platform.machine(),
            "transformers_version": None,
            "huggingface_hub_version": None,
        },
        network={
            "policy": "forbidden",
            "attempted": blocked_attempt_count > 0,
            "blocked_attempt_count": blocked_attempt_count,
            "enforcement": "python_stdlib_socket_guard",
            "enforcement_scope": (
                _LINUX_NETNS_SCOPE
                if os_isolation == _LINUX_NETNS
                else _PYTHON_GUARD_SCOPE
            ),
            "os_isolation": os_isolation,
        },
        implementation_sha256=_file_sha256(Path(__file__).resolve()),
        checks=tuple(checks),
        summary=summary,
        claim_boundary=_DSPARK_SUITE_BOUNDARY,
    )


def run_conformance(
    *,
    profile: str = "core",
    require_os_network_isolation: bool = False,
) -> ConformanceReport:
    """Run one fixed cumulative profile and return a canonical aggregate receipt."""

    if profile not in _PROFILE_CHECKS:
        choices = ", ".join(_PROFILE_CHECKS)
        raise ValueError(f"profile must be one of: {choices}.")
    if require_os_network_isolation and profile == "live":
        raise ValueError("live conformance cannot require no-network OS isolation")
    with _NETWORK_LOCK:
        return _run_conformance_locked(
            profile=profile,
            require_os_network_isolation=require_os_network_isolation,
        )


def _run_conformance_locked(
    *,
    profile: str,
    require_os_network_isolation: bool,
) -> ConformanceReport:
    os_isolation = _os_network_isolation()
    if require_os_network_isolation and os_isolation != _LINUX_NETNS:
        raise RuntimeError("required OS network isolation is not active")
    if profile == "dspark":
        return _run_dspark_profile_locked(os_isolation=os_isolation)

    selected = set(_PROFILE_CHECKS[profile])
    transformers_version: str | None = None
    huggingface_hub_version: str | None = None
    checks: list[ConformanceCheck] = []
    accounted_attempts = 0
    last_selected_offline_index: int | None = None
    with _offline_network_guard() as guard:
        for check_id in _CHECK_ORDER[:-1]:
            if check_id not in selected:
                check = _skip_check(
                    check_id,
                    required=False,
                    reason_code="not_selected_by_profile",
                )
            elif check_id == "tiny_model_cache":
                check = _run_demo_check()
            elif check_id == "flash_0731_architecture":
                check = _run_architecture_check()
            elif check_id == "attention_reach":
                check = _run_attention_check()
            else:
                try:
                    transformers_version = _distribution_version("transformers")
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception:
                    check = _error_check(check_id, required=True)
                else:
                    check = _run_parity_check(transformers_version)
            checks.append(check)
            if check_id in selected:
                last_selected_offline_index = len(checks) - 1
            if guard.blocked_attempt_count > accounted_attempts:
                target_index = (
                    len(checks) - 1
                    if check_id in selected
                    else last_selected_offline_index
                )
                if target_index is None:  # pragma: no cover - profiles select core
                    raise RuntimeError("network attempt preceded every selected check")
                checks[target_index] = _error_check(
                    checks[target_index].id,
                    required=True,
                    reason_code="network_policy_violation",
                )
                accounted_attempts = guard.blocked_attempt_count

    blocked_attempt_count = guard.blocked_attempt_count
    if blocked_attempt_count > accounted_attempts:
        if last_selected_offline_index is None:  # pragma: no cover - profiles select core
            raise RuntimeError("network attempt occurred without a selected offline check")
        checks[last_selected_offline_index] = _error_check(
            checks[last_selected_offline_index].id,
            required=True,
            reason_code="network_policy_violation",
        )

    live_attempted = False
    live_id = _CHECK_ORDER[-1]
    if live_id not in selected:
        checks.append(
            _skip_check(
                live_id,
                required=False,
                reason_code="not_selected_by_profile",
            )
        )
    else:
        try:
            huggingface_hub_version = _distribution_version("huggingface_hub")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            live_check = _error_check(live_id, required=True)
        else:
            live_check, live_attempted = _run_live_check(huggingface_hub_version)
        checks.append(live_check)

    status = _suite_status(checks)
    summary = {
        state: sum(check.status == state for check in checks)
        for state in ("pass", "fail", "skip", "error")
    }
    summary["required"] = sum(check.required for check in checks)
    complete = all(
        not check.required or check.status in {"pass", "fail"}
        for check in checks
    )
    return ConformanceReport(
        schema_version=_SCHEMA_VERSION,
        kind="nano-deepseek-v4-conformance",
        profile=profile,
        status=status,
        passed=status == "pass",
        complete=complete,
        package={"name": "nano-deepseek-v4", "version": __version__},
        environment={
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "platform": platform.system(),
            "machine": platform.machine(),
            "transformers_version": transformers_version,
            "huggingface_hub_version": huggingface_hub_version,
        },
        network={
            "policy": "pinned_hub_metadata_only" if profile == "live" else "forbidden",
            "attempted": blocked_attempt_count > 0 or live_attempted,
            "blocked_attempt_count": blocked_attempt_count,
            "enforcement": (
                "python_stdlib_socket_guard+verifier_callback"
                if profile == "live"
                else "python_stdlib_socket_guard"
            ),
            "enforcement_scope": (
                _LINUX_NETNS_SCOPE
                if os_isolation == _LINUX_NETNS
                else _PYTHON_GUARD_SCOPE
            ),
            "os_isolation": os_isolation,
        },
        implementation_sha256=_file_sha256(Path(__file__).resolve()),
        checks=tuple(checks),
        summary=summary,
        claim_boundary=_SUITE_BOUNDARY,
    )


def _json_payload(report: ConformanceReport) -> str:
    return json.dumps(
        report.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _print_human(report: ConformanceReport) -> None:
    print(f"nano-deepseek-v4 conformance ({report.profile})")
    for check in report.checks:
        suffix = ""
        if check.reason_code is not None:
            suffix = f" — {_REASON_MESSAGES.get(check.reason_code, check.reason_code)}"
        requirement = "required" if check.required else "optional"
        print(f"{check.status.upper():<10} {check.id} [{requirement}]{suffix}")
    print(f"{report.profile}: {report.status.upper()}")
    print(f"scope: {report.claim_boundary}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed nano-deepseek-v4 conformance profile.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(_PROFILE_CHECKS),
        default="core",
        help=(
            "core is local; dspark adds fixed semantic vectors; offline adds pinned "
            "Transformers; live also replays Hub metadata."
        ),
    )
    parser.add_argument(
        "--require-os-network-isolation",
        action="store_true",
        help="Require a verified Linux no-network namespace (core/dspark/offline only).",
    )
    parser.add_argument("--json", action="store_true", help="Emit one canonical JSON report.")
    parser.add_argument("--output", type=Path, help="Atomically write the JSON report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_conformance(
            profile=args.profile,
            require_os_network_isolation=args.require_os_network_isolation,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"conformance run failed: {type(exc).__name__}", file=sys.stderr)
        return 4
    try:
        payload = _json_payload(report)
    except (TypeError, ValueError) as exc:
        print(f"conformance report serialization failed: {type(exc).__name__}", file=sys.stderr)
        return 4

    output_failed = False
    if args.output is not None:
        try:
            _write_atomic(args.output, payload)
        except OSError as exc:
            print(f"could not write conformance report: {type(exc).__name__}", file=sys.stderr)
            output_failed = True
    if args.json:
        print(payload, end="")
    else:
        _print_human(report)
    return 4 if output_failed else _EXIT_CODES[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
