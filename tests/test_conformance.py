from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

import nano_deepseek_v4.conformance as conformance
from nano_deepseek_v4.conformance import (
    ConformanceCheck,
    ConformanceReport,
    _json_payload,
    main,
    run_conformance,
)
from nano_deepseek_v4.verify_flash_0731_receipt import (
    ReceiptUnavailableError,
    ReceiptUnavailableReason,
    ReceiptVerificationError,
)


@dataclass(frozen=True)
class _FakeReceipt:
    kind: str
    passed: bool = True
    claim_boundary: str = "fixture-only claim boundary"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": self.kind,
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


class _FakeHuggingFaceConstants(ModuleType):
    HF_HUB_OFFLINE: bool
    HF_HUB_DISABLE_TELEMETRY: bool

    def __init__(self, *, offline: bool, telemetry: bool):
        super().__init__("huggingface_hub.constants")
        self.HF_HUB_OFFLINE = offline
        self.HF_HUB_DISABLE_TELEMETRY = telemetry


class _FakeHuggingFaceRequestsHttp(ModuleType):
    def __init__(self, callback: Callable[[], None]):
        super().__init__("huggingface_hub.utils._http")
        self._callback = callback

    def reset_sessions(self) -> None:
        self._callback()


class _FakeHuggingFaceHttpxHttp(ModuleType):
    def __init__(self, callback: Callable[[], None]):
        super().__init__("huggingface_hub.utils._http")
        self._callback = callback

    def close_session(self) -> None:
        self._callback()


def _patch_versions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transformers: str | None = "5.15.0",
    huggingface_hub: str | None = "1.27.0",
) -> None:
    versions = {
        "transformers": transformers,
        "huggingface_hub": huggingface_hub,
    }
    monkeypatch.setattr(conformance, "_distribution_version", versions.__getitem__)


def _network_receipt(
    profile: str,
    *,
    attempted: bool,
    blocked_attempt_count: int = 0,
    os_isolation: str = "none",
) -> dict[str, str | bool | int]:
    return {
        "policy": "pinned_hub_metadata_only" if profile == "live" else "forbidden",
        "attempted": attempted,
        "blocked_attempt_count": blocked_attempt_count,
        "enforcement": (
            "python_stdlib_socket_guard+verifier_callback"
            if profile == "live"
            else "python_stdlib_socket_guard"
        ),
        "enforcement_scope": (
            "linux_network_namespace_process_tree"
            if os_isolation == "linux_network_namespace"
            else "best_effort_python_stdlib_calls"
        ),
        "os_isolation": os_isolation,
    }


def _patch_passing_runners(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str] | None = None,
) -> None:
    observed = calls if calls is not None else []

    def run_demo() -> _FakeReceipt:
        observed.append("tiny_model_cache")
        return _FakeReceipt("tiny-model-cache")

    def architecture_receipt() -> tuple[dict[str, Any], bool]:
        observed.append("flash_0731_architecture")
        return {
            "schema_version": 1,
            "kind": "flash-0731-architecture",
            "claim_boundary": "fixture-only architecture boundary",
        }, True

    def run_attention_reach(*, device: str) -> _FakeReceipt:
        assert device == "cpu"
        observed.append("attention_reach")
        return _FakeReceipt("attention-reach")

    def run_dspark_vector_conformance() -> _FakeReceipt:
        observed.append("dspark_vectors")
        return _FakeReceipt("dspark-semantic-conformance")

    def run_dspark_scheduler_conformance() -> _FakeReceipt:
        observed.append("dspark_scheduler")
        return _FakeReceipt("dspark-scheduler-conformance")

    def run_transformers_parity() -> _FakeReceipt:
        observed.append("transformers_parity")
        return _FakeReceipt("transformers-parity")

    def verify_flash_0731_receipt(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> dict[str, Any]:
        assert cache_dir.is_dir()
        on_network_attempt()
        observed.append("flash_0731_receipt")
        return {
            "schema_version": 1,
            "kind": "flash-0731-receipt",
            "status": "pass",
            "scope_boundary": "fixture-only live boundary",
        }

    monkeypatch.setattr(conformance, "run_demo", run_demo)
    monkeypatch.setattr(conformance, "_architecture_receipt", architecture_receipt)
    monkeypatch.setattr(conformance, "run_attention_reach", run_attention_reach)
    monkeypatch.setattr(
        conformance,
        "run_dspark_vector_conformance",
        run_dspark_vector_conformance,
    )
    monkeypatch.setattr(
        conformance,
        "run_dspark_scheduler_conformance",
        run_dspark_scheduler_conformance,
    )
    monkeypatch.setattr(conformance, "run_transformers_parity", run_transformers_parity)
    monkeypatch.setattr(
        conformance,
        "verify_flash_0731_receipt",
        verify_flash_0731_receipt,
    )


@pytest.mark.parametrize(
    ("profile", "executed", "statuses", "required", "network_attempted"),
    [
        (
            "core",
            ["tiny_model_cache", "flash_0731_architecture", "attention_reach"],
            ["pass", "pass", "pass", "skip", "skip"],
            [True, True, True, False, False],
            False,
        ),
        (
            "offline",
            [
                "tiny_model_cache",
                "flash_0731_architecture",
                "attention_reach",
                "transformers_parity",
            ],
            ["pass", "pass", "pass", "pass", "skip"],
            [True, True, True, True, False],
            False,
        ),
        (
            "live",
            [
                "tiny_model_cache",
                "flash_0731_architecture",
                "attention_reach",
                "transformers_parity",
                "flash_0731_receipt",
            ],
            ["pass", "pass", "pass", "pass", "pass"],
            [True, True, True, True, True],
            True,
        ),
    ],
)
def test_profiles_have_fixed_cumulative_composition_and_order(
    profile: str,
    executed: list[str],
    statuses: list[str],
    required: list[bool],
    network_attempted: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch, calls)

    report = run_conformance(profile=profile)

    assert calls == executed
    assert [check.id for check in report.checks] == [
        "tiny_model_cache",
        "flash_0731_architecture",
        "attention_reach",
        "transformers_parity",
        "flash_0731_receipt",
    ]
    assert [check.status for check in report.checks] == statuses
    assert [check.required for check in report.checks] == required
    assert report.status == "pass"
    assert report.passed is True
    assert report.complete is True
    assert report.network == _network_receipt(profile, attempted=network_attempted)


def test_dspark_profile_is_fixed_offline_and_does_not_query_optional_versions(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    _patch_passing_runners(monkeypatch, calls)

    def forbidden(name: str) -> str:
        raise AssertionError(f"dspark queried optional distribution metadata: {name}")

    monkeypatch.setattr(conformance, "_distribution_version", forbidden)

    report = run_conformance(profile="dspark")

    assert calls == [
        "tiny_model_cache",
        "flash_0731_architecture",
        "attention_reach",
        "dspark_vectors",
        "dspark_scheduler",
    ]
    assert [check.id for check in report.checks] == [
        "tiny_model_cache",
        "flash_0731_architecture",
        "attention_reach",
        "dspark_vectors",
        "dspark_scheduler",
        "transformers_parity",
        "flash_0731_receipt",
    ]
    assert [check.status for check in report.checks] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "skip",
        "skip",
    ]
    assert [check.required for check in report.checks] == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert report.status == "pass"
    assert report.complete is True
    assert report.network == _network_receipt("dspark", attempted=False)
    assert report.environment["transformers_version"] is None
    assert report.environment["huggingface_hub_version"] is None


@pytest.mark.parametrize(
    ("runner", "check_status", "suite_status"),
    [
        (lambda: _FakeReceipt("dspark", passed=False), "fail", "fail"),
        (lambda: (_ for _ in ()).throw(PermissionError("/private/vector")), "error", "error"),
    ],
)
def test_dspark_profile_surfaces_mismatch_and_sanitizes_harness_errors(
    runner: Callable[[], _FakeReceipt],
    check_status: str,
    suite_status: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_passing_runners(monkeypatch)
    monkeypatch.setattr(conformance, "run_dspark_vector_conformance", runner)

    report = run_conformance(profile="dspark")

    check = report.checks[3]
    assert check.id == "dspark_vectors"
    assert check.status == check_status
    assert report.status == suite_status
    assert "/private/vector" not in _json_payload(report)


@pytest.mark.parametrize(
    ("runner", "check_status", "suite_status"),
    [
        (lambda: _FakeReceipt("dspark-scheduler", passed=False), "fail", "fail"),
        (lambda: (_ for _ in ()).throw(PermissionError("/private/sps")), "error", "error"),
    ],
)
def test_dspark_profile_surfaces_scheduler_mismatch_and_sanitizes_errors(
    runner: Callable[[], _FakeReceipt],
    check_status: str,
    suite_status: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_passing_runners(monkeypatch)
    monkeypatch.setattr(conformance, "run_dspark_scheduler_conformance", runner)

    report = run_conformance(profile="dspark")

    check = report.checks[4]
    assert check.id == "dspark_scheduler"
    assert check.status == check_status
    assert report.status == suite_status
    assert "/private/sps" not in _json_payload(report)


def test_core_never_runs_optional_or_network_checks(monkeypatch: pytest.MonkeyPatch):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)

    def forbidden() -> None:
        raise AssertionError("optional runner must not execute in the core profile")

    def forbidden_live(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> None:
        del cache_dir, on_network_attempt
        raise AssertionError("network runner must not execute in the core profile")

    monkeypatch.setattr(conformance, "run_transformers_parity", forbidden)
    monkeypatch.setattr(conformance, "verify_flash_0731_receipt", forbidden_live)

    report = run_conformance(profile="core")

    assert report.status == "pass"
    assert report.network == _network_receipt("core", attempted=False)
    assert [check.reason_code for check in report.checks[3:]] == [
        "not_selected_by_profile",
        "not_selected_by_profile",
    ]


def test_core_does_not_query_optional_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_passing_runners(monkeypatch)

    def forbidden(name: str) -> str:
        raise AssertionError(f"core queried optional distribution metadata: {name}")

    monkeypatch.setattr(conformance, "_distribution_version", forbidden)

    report = run_conformance(profile="core")

    assert report.status == "pass"
    assert report.environment["transformers_version"] is None
    assert report.environment["huggingface_hub_version"] is None


@pytest.mark.parametrize(
    ("profile", "broken_distribution", "check_index"),
    [("offline", "transformers", 3), ("live", "huggingface_hub", 4)],
)
def test_selected_distribution_metadata_errors_are_sanitized_check_errors(
    profile: str,
    broken_distribution: str,
    check_index: int,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_passing_runners(monkeypatch)

    def version(name: str) -> str:
        if name == broken_distribution:
            raise PermissionError(f"/private/{name}.dist-info")
        return {"transformers": "5.15.0", "huggingface_hub": "1.27.0"}[name]

    monkeypatch.setattr(conformance, "_distribution_version", version)

    report = run_conformance(profile=profile)

    check = report.checks[check_index]
    assert check.status == "error"
    assert check.reason_code == "unexpected_exception"
    assert report.status == "error"
    assert "/private/" not in _json_payload(report)


@pytest.mark.parametrize("operation", ["tcp", "udp", "dns", "saved_socket"])
def test_offline_guard_fails_even_when_a_runner_swallows_the_block(
    operation: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)
    saved_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def network_attempt() -> _FakeReceipt:
        try:
            if operation == "tcp":
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            elif operation == "udp":
                socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif operation == "dns":
                socket.getaddrinfo("example.invalid", 443)
            else:
                saved_socket.send(b"forbidden")
        except Exception:
            pass
        return _FakeReceipt("tiny-model-cache")

    monkeypatch.setattr(conformance, "run_demo", network_attempt)
    try:
        report = run_conformance(profile="core")
    finally:
        saved_socket.close()

    check = report.checks[0]
    assert check.status == "error"
    assert check.reason_code == "network_policy_violation"
    assert report.status == "error"
    assert report.complete is False
    assert report.network == _network_receipt(
        "core",
        attempted=True,
        blocked_attempt_count=1,
    )


@pytest.mark.parametrize("gap_after_index", [0, 3])
def test_offline_guard_accounts_for_attempts_between_checks_and_at_finalization(
    gap_after_index: int,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)
    canonical_order = tuple(conformance._CHECK_ORDER)

    class GapOrder:
        def __getitem__(self, index: int | slice) -> str | object:
            if isinstance(index, slice):
                values = canonical_order[index]

                def iterate_with_gap():
                    for item_index, item in enumerate(values):
                        yield item
                        if item_index == gap_after_index:
                            try:
                                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            except Exception:
                                pass

                return iterate_with_gap()
            return canonical_order[index]

    monkeypatch.setattr(conformance, "_CHECK_ORDER", GapOrder())

    report = run_conformance(profile="core")

    assert report.status == "error"
    assert report.complete is False
    assert any(
        check.required
        and check.status == "error"
        and check.reason_code == "network_policy_violation"
        for check in report.checks
    )
    assert report.network == _network_receipt(
        "core",
        attempted=True,
        blocked_attempt_count=1,
    )


def test_offline_guard_allows_unix_sockets_and_restores_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)

    def local_runner() -> _FakeReceipt:
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        left, right = socket.socketpair()
        try:
            left.sendall(b"ok")
            assert right.recv(2) == b"ok"
        finally:
            left.close()
            right.close()
        return _FakeReceipt("tiny-model-cache")

    monkeypatch.setattr(conformance, "run_demo", local_runner)
    report = run_conformance(profile="core")

    assert report.status == "pass"
    assert report.network == _network_receipt("core", attempted=False)
    assert os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert "HF_HUB_DISABLE_TELEMETRY" not in os.environ


def test_live_verifier_runs_after_the_offline_socket_guard_is_removed(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)

    def live_verifier(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> dict[str, Any]:
        assert cache_dir.is_dir()
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.close()
        on_network_attempt()
        return {
            "schema_version": 1,
            "status": "pass",
            "scope_boundary": "fixture-only live boundary",
        }

    monkeypatch.setattr(conformance, "verify_flash_0731_receipt", live_verifier)

    report = run_conformance(profile="live")

    assert report.status == "pass"
    assert report.network == _network_receipt("live", attempted=True)


def test_offline_guard_synchronizes_preimported_hub_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    constants = _FakeHuggingFaceConstants(offline=False, telemetry=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        monkeypatch.delenv(name, raising=False)

    with conformance._offline_network_guard():
        assert constants.HF_HUB_OFFLINE is True
        assert constants.HF_HUB_DISABLE_TELEMETRY is True

    assert constants.HF_HUB_OFFLINE is False
    assert constants.HF_HUB_DISABLE_TELEMETRY is False


def test_offline_guard_restores_hub_flags_imported_inside_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        monkeypatch.delenv(name, raising=False)

    with conformance._offline_network_guard():
        constants = _FakeHuggingFaceConstants(offline=True, telemetry=True)
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)

    assert constants.HF_HUB_OFFLINE is False
    assert constants.HF_HUB_DISABLE_TELEMETRY is False


def test_offline_guard_preserves_inherited_hub_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    constants = _FakeHuggingFaceConstants(offline=False, telemetry=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "yes")
    monkeypatch.setenv("DO_NOT_TRACK", "ON")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    monkeypatch.delenv("DISABLE_TELEMETRY", raising=False)

    with conformance._offline_network_guard():
        assert constants.HF_HUB_OFFLINE is True
        assert constants.HF_HUB_DISABLE_TELEMETRY is True

    assert constants.HF_HUB_OFFLINE is True
    assert constants.HF_HUB_DISABLE_TELEMETRY is True


def test_offline_guard_matches_hub_flag_semantics_for_explicit_false(
    monkeypatch: pytest.MonkeyPatch,
):
    constants = _FakeHuggingFaceConstants(offline=False, telemetry=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "false")
    monkeypatch.setenv("DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("DO_NOT_TRACK", "true")

    with conformance._offline_network_guard():
        assert constants.HF_HUB_OFFLINE is True
        assert constants.HF_HUB_DISABLE_TELEMETRY is True

    assert constants.HF_HUB_OFFLINE is False
    assert constants.HF_HUB_DISABLE_TELEMETRY is True


def test_offline_guard_resets_cached_hub_requests_sessions(
    monkeypatch: pytest.MonkeyPatch,
):
    constants = _FakeHuggingFaceConstants(offline=False, telemetry=False)
    observed_offline_states: list[bool] = []
    http = _FakeHuggingFaceRequestsHttp(
        lambda: observed_offline_states.append(constants.HF_HUB_OFFLINE)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._http", http)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with conformance._offline_network_guard():
        assert observed_offline_states == [True]

    assert observed_offline_states == [True, False]


def test_offline_guard_closes_cached_hub_httpx_session(
    monkeypatch: pytest.MonkeyPatch,
):
    constants = _FakeHuggingFaceConstants(offline=False, telemetry=False)
    observed_offline_states: list[bool] = []
    http = _FakeHuggingFaceHttpxHttp(
        lambda: observed_offline_states.append(constants.HF_HUB_OFFLINE)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._http", http)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with conformance._offline_network_guard():
        assert observed_offline_states == [True]

    assert observed_offline_states == [True, False]


def test_offline_guard_prefers_requests_session_reset_when_both_apis_exist(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    constants = _FakeHuggingFaceConstants(offline=False, telemetry=False)
    http = _FakeHuggingFaceRequestsHttp(lambda: events.append("reset_sessions"))
    monkeypatch.setattr(
        http,
        "close_session",
        lambda: events.append("close_session"),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._http", http)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with conformance._offline_network_guard():
        pass

    assert events == ["reset_sessions", "reset_sessions"]


def test_offline_guard_releases_active_state_after_session_reset_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    constants = _FakeHuggingFaceConstants(offline=False, telemetry=False)
    reset_count = 0

    def reset_session() -> None:
        nonlocal reset_count
        reset_count += 1
        if reset_count == 2:
            raise RuntimeError("fixture exit reset failure")

    http = _FakeHuggingFaceRequestsHttp(reset_session)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils._http", http)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with pytest.raises(RuntimeError, match="fixture exit reset failure"):
        with conformance._offline_network_guard():
            pass

    assert conformance._ACTIVE_NETWORK_GUARD is None
    with conformance._offline_network_guard():
        assert conformance._ACTIVE_NETWORK_GUARD is not None

    assert reset_count == 4


def test_live_profile_restores_hub_state_imported_by_offline_parity(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.utils._http", raising=False)
    for name in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "DISABLE_TELEMETRY",
        "DO_NOT_TRACK",
    ):
        monkeypatch.delenv(name, raising=False)

    loaded_constants: list[_FakeHuggingFaceConstants] = []
    reset_states: list[bool] = []

    def parity_runner() -> _FakeReceipt:
        constants = _FakeHuggingFaceConstants(offline=True, telemetry=True)
        http = _FakeHuggingFaceRequestsHttp(
            lambda: reset_states.append(constants.HF_HUB_OFFLINE)
        )
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
        monkeypatch.setitem(sys.modules, "huggingface_hub.utils._http", http)
        loaded_constants.append(constants)
        return _FakeReceipt("transformers-parity")

    def live_verifier(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> dict[str, Any]:
        assert cache_dir.is_dir()
        assert loaded_constants[0].HF_HUB_OFFLINE is False
        assert loaded_constants[0].HF_HUB_DISABLE_TELEMETRY is False
        assert reset_states == [False]
        on_network_attempt()
        return {
            "schema_version": 1,
            "status": "pass",
            "scope_boundary": "fixture-only live boundary",
        }

    monkeypatch.setattr(conformance, "run_transformers_parity", parity_runner)
    monkeypatch.setattr(conformance, "verify_flash_0731_receipt", live_verifier)

    report = run_conformance(profile="live")

    assert report.status == "pass"
    assert report.network == _network_receipt("live", attempted=True)


@pytest.mark.parametrize("previous_cache", [None, "/caller/torchinductor-cache"])
def test_parity_check_restores_rng_and_contains_inductor_cache(
    previous_cache: str | None,
    monkeypatch: pytest.MonkeyPatch,
):
    observed_cache_dirs: list[Path] = []

    def parity_runner() -> _FakeReceipt:
        random.random()
        np.random.random_sample()
        torch.rand(())
        cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "lazy-import-marker").write_text("fixture", encoding="utf-8")
        observed_cache_dirs.append(cache_dir)
        return _FakeReceipt("transformers-parity")

    monkeypatch.setattr(conformance, "run_transformers_parity", parity_runner)
    if previous_cache is None:
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", previous_cache)

    random.seed(12_345)
    expected_python = random.random()
    random.seed(12_345)
    np.random.seed(12_345)
    expected_numpy = float(np.random.random_sample())
    np.random.seed(12_345)
    torch.manual_seed(12_345)
    expected_torch = torch.rand(())
    torch.manual_seed(12_345)

    check = conformance._run_parity_check("5.15.0")

    assert check.status == "pass"
    assert random.random() == expected_python
    assert float(np.random.random_sample()) == expected_numpy
    assert torch.equal(torch.rand(()), expected_torch)
    assert os.environ.get("TORCHINDUCTOR_CACHE_DIR") == previous_cache
    assert len(observed_cache_dirs) == 1
    assert not observed_cache_dirs[0].exists()


def test_parity_check_restores_process_state_after_runner_error(
    monkeypatch: pytest.MonkeyPatch,
):
    observed_cache_dirs: list[Path] = []

    def parity_runner() -> _FakeReceipt:
        random.random()
        np.random.random_sample()
        torch.rand(())
        cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        cache_dir.mkdir(parents=True, exist_ok=True)
        observed_cache_dirs.append(cache_dir)
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(conformance, "run_transformers_parity", parity_runner)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    random.seed(54_321)
    python_state = random.getstate()
    np.random.seed(54_321)
    numpy_state = np.random.get_state()
    torch.manual_seed(54_321)
    torch_state = torch.random.get_rng_state().clone()

    check = conformance._run_parity_check("5.15.0")

    assert check.status == "error"
    assert random.getstate() == python_state
    restored_numpy_state = np.random.get_state()
    assert restored_numpy_state[0] == numpy_state[0]
    assert np.array_equal(restored_numpy_state[1], numpy_state[1])
    assert restored_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
    assert len(observed_cache_dirs) == 1
    assert not observed_cache_dirs[0].exists()


def test_real_cold_parity_to_live_restores_process_state(tmp_path: Path):
    try:
        transformers_version = metadata.version("transformers")
        metadata.version("huggingface_hub")
    except metadata.PackageNotFoundError:
        pytest.skip("cold parity integration requires the parity and official extras")
    if transformers_version != "5.15.0":
        pytest.skip("cold parity integration requires Transformers 5.15.0")

    script = r"""
import json
import os
import random

import numpy as np
import torch

import nano_deepseek_v4.conformance as conformance

assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
random.seed(12345)
expected_python = random.random()
random.seed(12345)
np.random.seed(12345)
expected_numpy = float(np.random.random_sample())
np.random.seed(12345)
torch.manual_seed(12345)
generator = torch.Generator().manual_seed(12345)
expected_torch = torch.rand((), generator=generator)


def live_verifier(*, cache_dir, on_network_attempt):
    import huggingface_hub.constants as constants

    assert cache_dir.is_dir()
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "false"
    assert os.environ["DISABLE_TELEMETRY"] == "true"
    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
    assert constants.HF_HUB_OFFLINE is False
    assert constants.HF_HUB_DISABLE_TELEMETRY is True
    on_network_attempt()
    return {
        "schema_version": 1,
        "status": "pass",
        "scope_boundary": "cold-import fixture-only live boundary",
    }


conformance.verify_flash_0731_receipt = live_verifier
report = conformance.run_conformance(profile="live")
assert report.status == "pass"
assert random.random() == expected_python
assert float(np.random.random_sample()) == expected_numpy
assert torch.equal(torch.rand(()), expected_torch)
assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
print(json.dumps({"status": report.status, "attempted": report.network["attempted"]}))
"""
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(Path(conformance.__file__).resolve().parents[1]),
            "TMPDIR": str(tmp_path),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "false",
            "DISABLE_TELEMETRY": "true",
        }
    )
    environment.pop("TORCHINDUCTOR_CACHE_DIR", None)

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"attempted": True, "status": "pass"}
    assert list(tmp_path.glob("nano-dsv4-parity-cache-*")) == []
    assert list(tmp_path.glob("torchinductor_*")) == []


def test_required_os_network_isolation_is_fail_closed_and_recorded(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)
    monkeypatch.setattr(conformance, "_os_network_isolation", lambda: "none")

    with pytest.raises(RuntimeError, match="required OS network isolation"):
        run_conformance(profile="core", require_os_network_isolation=True)
    assert main(
        ["--profile", "core", "--require-os-network-isolation", "--json"]
    ) == 4
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "conformance run failed: RuntimeError\n"

    monkeypatch.setattr(
        conformance,
        "_os_network_isolation",
        lambda: "linux_network_namespace",
    )
    report = run_conformance(profile="core", require_os_network_isolation=True)
    assert report.status == "pass"
    assert report.network == _network_receipt(
        "core",
        attempted=False,
        os_isolation="linux_network_namespace",
    )


def test_os_network_isolation_reads_current_namespace_interfaces(
    monkeypatch: pytest.MonkeyPatch,
):
    status = """\
CapInh:\t0000000000000000
CapPrm:\t0000000000000000
CapEff:\t0000000000000000
CapBnd:\t0000000000000000
CapAmb:\t0000000000000000
NoNewPrivs:\t1
"""
    monkeypatch.setenv(conformance._OS_ISOLATION_ENV, conformance._LINUX_NETNS)
    monkeypatch.setenv(conformance._PARENT_NETNS_ENV, "net:[1]")
    monkeypatch.setenv(conformance._EXPECTED_UID_ENV, "1001")
    monkeypatch.setenv(conformance._EXPECTED_GID_ENV, "1002")
    monkeypatch.setattr(conformance.platform, "system", lambda: "Linux")
    monkeypatch.setattr(conformance.os, "readlink", lambda _path: "net:[2]")
    monkeypatch.setattr(conformance.os, "geteuid", lambda: 1001)
    monkeypatch.setattr(conformance.os, "getegid", lambda: 1002)
    monkeypatch.setattr(conformance.os, "getgroups", lambda: [])
    monkeypatch.setattr(conformance.Path, "read_text", lambda *_args, **_kwargs: status)
    monkeypatch.setattr(conformance.socket, "if_nameindex", lambda: [(1, "lo")])

    assert conformance._os_network_isolation() == conformance._LINUX_NETNS

    monkeypatch.setattr(
        conformance.socket,
        "if_nameindex",
        lambda: [(1, "lo"), (2, "eth0")],
    )
    assert conformance._os_network_isolation() == "none"


def test_process_isolation_receipt_requires_dropped_privileges():
    status = """\
CapInh:\t0000000000000000
CapPrm:\t0000000000000000
CapEff:\t0000000000000000
CapBnd:\t0000000000000000
CapAmb:\t0000000000000000
NoNewPrivs:\t1
"""
    def verified(candidate: str, *, groups: set[int] | None = None) -> bool:
        return conformance._process_isolation_verified(
            candidate,
            expected_uid=1001,
            expected_gid=1002,
            current_uid=1001,
            current_gid=1002,
            supplementary_groups=set() if groups is None else groups,
        )

    assert verified(status) is True
    assert (
        verified(status.replace("CapBnd:\t0000000000000000", "CapBnd:\t1"))
        is False
    )
    assert (
        verified(status.replace("NoNewPrivs:\t1", "NoNewPrivs:\t0"))
        is False
    )
    assert verified(status, groups={27}) is False


@pytest.mark.parametrize(
    ("installed_version", "reason_code"),
    [
        (None, "missing_optional_dependency"),
        ("5.14.0", "incompatible_optional_dependency"),
    ],
)
def test_offline_requires_the_exact_transformers_reference(
    installed_version: str | None,
    reason_code: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch, transformers=installed_version)
    _patch_passing_runners(monkeypatch)

    def forbidden() -> None:
        raise AssertionError("parity runner must not execute without the pinned dependency")

    monkeypatch.setattr(conformance, "run_transformers_parity", forbidden)

    report = run_conformance(profile="offline")
    parity = report.checks[3]

    assert parity.id == "transformers_parity"
    assert parity.required is True
    assert parity.status == "skip"
    assert parity.reason_code == reason_code
    assert parity.receipt is None
    assert report.status == "incomplete"
    assert report.passed is False
    assert report.complete is False


@pytest.mark.parametrize(
    "reason_code",
    ["source_unavailable", "inspection_unavailable"],
)
def test_live_typed_unavailability_is_incomplete_not_drift(
    reason_code: ReceiptUnavailableReason,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)

    def unavailable(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> None:
        del cache_dir
        on_network_attempt()
        raise ReceiptUnavailableError("temporary outage", reason_code=reason_code)

    monkeypatch.setattr(conformance, "verify_flash_0731_receipt", unavailable)

    report = run_conformance(profile="live")
    live = report.checks[4]

    assert live.status == "skip"
    assert live.reason_code == reason_code
    assert live.receipt is None
    assert report.status == "incomplete"
    assert report.passed is False
    assert report.complete is False
    assert report.network["attempted"] is True


def test_live_typed_semantic_drift_is_a_failed_check(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)

    def drifted(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> None:
        del cache_dir
        on_network_attempt()
        raise ReceiptVerificationError("pinned source drift")

    monkeypatch.setattr(conformance, "verify_flash_0731_receipt", drifted)

    report = run_conformance(profile="live")
    live = report.checks[4]

    assert live.status == "fail"
    assert live.reason_code == "receipt_drift"
    assert live.receipt is None
    assert report.status == "fail"
    assert report.complete is True
    assert report.network["attempted"] is True


def test_live_missing_hub_dependency_does_not_attempt_network(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch, huggingface_hub=None)
    _patch_passing_runners(monkeypatch)

    def forbidden(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> None:
        del cache_dir, on_network_attempt
        raise AssertionError("live runner must not execute without its dependency")

    monkeypatch.setattr(conformance, "verify_flash_0731_receipt", forbidden)

    report = run_conformance(profile="live")
    live = report.checks[4]

    assert live.status == "skip"
    assert live.reason_code == "missing_optional_dependency"
    assert report.status == "incomplete"
    assert report.network["attempted"] is False


@pytest.mark.parametrize("hub_version", ["0.33.5", "0.34.0.dev1"])
def test_live_requires_a_supported_hub_client(
    hub_version: str,
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch, huggingface_hub=hub_version)
    _patch_passing_runners(monkeypatch)

    def forbidden(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> None:
        del cache_dir, on_network_attempt
        raise AssertionError("live runner must not execute with an old Hub client")

    monkeypatch.setattr(conformance, "verify_flash_0731_receipt", forbidden)

    report = run_conformance(profile="live")
    live = report.checks[4]

    assert live.status == "skip"
    assert live.reason_code == "incompatible_optional_dependency"
    assert report.status == "incomplete"
    assert report.network["attempted"] is False


def test_live_tempdir_failure_before_verifier_is_not_a_network_attempt(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)
    verifier_calls = 0

    def forbidden_verifier(
        *,
        cache_dir: Path,
        on_network_attempt: Callable[[], None],
    ) -> None:
        nonlocal verifier_calls
        del cache_dir, on_network_attempt
        verifier_calls += 1

    def tempdir_failure(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise PermissionError("/private/temp-root")

    monkeypatch.setattr(
        conformance,
        "verify_flash_0731_receipt",
        forbidden_verifier,
    )
    monkeypatch.setattr(conformance.tempfile, "TemporaryDirectory", tempdir_failure)

    report = run_conformance(profile="live")
    live = report.checks[4]

    assert verifier_calls == 0
    assert live.status == "error"
    assert live.reason_code == "unexpected_exception"
    assert report.status == "error"
    assert report.network == _network_receipt("live", attempted=False)
    assert "/private/temp-root" not in _json_payload(report)


def test_unexpected_runner_exception_is_an_error_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_versions(monkeypatch)
    _patch_passing_runners(monkeypatch)

    def broken(*, device: str) -> None:
        del device
        raise RuntimeError("sensitive local path /tmp/private")

    monkeypatch.setattr(conformance, "run_attention_reach", broken)

    report = run_conformance(profile="core")
    check = report.checks[2]

    assert check.status == "error"
    assert check.reason_code == "unexpected_exception"
    assert check.receipt is None
    assert report.status == "error"
    assert report.complete is False
    assert "/tmp/private" not in _json_payload(report)


def test_cli_sanitizes_conformance_setup_exceptions(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    def broken_setup(
        *,
        profile: str,
        require_os_network_isolation: bool,
    ) -> ConformanceReport:
        assert profile == "core"
        assert require_os_network_isolation is False
        raise RuntimeError("/private/metadata/path")

    monkeypatch.setattr(conformance, "run_conformance", broken_setup)

    assert main(["--profile", "core", "--json"]) == 4

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "conformance run failed: RuntimeError\n"
    assert "Traceback" not in captured.err
    assert "/private/metadata/path" not in captured.err


def _report_with_status(status: conformance.SuiteStatus) -> ConformanceReport:
    check_status: conformance.CheckStatus
    reason_code: str | None
    if status == "pass":
        check_status, reason_code = "pass", None
    elif status == "fail":
        check_status, reason_code = "fail", "contract_failed"
    elif status == "incomplete":
        check_status, reason_code = "skip", "missing_optional_dependency"
    else:
        check_status, reason_code = "error", "unexpected_exception"
    check = ConformanceCheck(
        id="tiny_model_cache",
        required=True,
        mode="offline_local",
        status=check_status,
        reason_code=reason_code,
        claim_boundary="fixture-only check boundary",
        receipt_schema_version=None,
        receipt_sha256=None,
        receipt=None,
    )
    return ConformanceReport(
        schema_version=1,
        kind="nano-deepseek-v4-conformance",
        profile="core",
        status=status,
        passed=status == "pass",
        complete=status in {"pass", "fail"},
        package={"name": "nano-deepseek-v4", "version": "fixture"},
        environment={
            "python_implementation": "CPython",
            "python_version": "fixture",
            "torch_version": "fixture",
            "platform": "fixture",
            "machine": "fixture",
            "transformers_version": None,
            "huggingface_hub_version": None,
        },
        network=_network_receipt("core", attempted=False),
        implementation_sha256="0" * 64,
        checks=(check,),
        summary={
            "pass": int(check_status == "pass"),
            "fail": int(check_status == "fail"),
            "skip": int(check_status == "skip"),
            "error": int(check_status == "error"),
            "required": 1,
        },
        claim_boundary="fixture-only suite boundary",
    )


@pytest.mark.parametrize(
    ("status", "exit_code"),
    [("pass", 0), ("fail", 1), ("incomplete", 3), ("error", 4)],
)
def test_cli_exit_codes_and_writes_json_even_on_nonzero(
    status: conformance.SuiteStatus,
    exit_code: int,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    report = _report_with_status(status)
    monkeypatch.setattr(
        conformance,
        "run_conformance",
        lambda *, profile, require_os_network_isolation: report,
    )
    output = tmp_path / f"{status}.json"

    assert main(["--profile", "core", "--json", "--output", str(output)]) == exit_code

    stdout = capsys.readouterr().out
    output_bytes = output.read_bytes()
    assert stdout.encode("utf-8") == output_bytes
    assert output_bytes == _json_payload(report).encode("utf-8")
    assert output_bytes.endswith(b"\n")
    assert not output_bytes.endswith(b"\n\n")
    assert json.loads(output_bytes)["status"] == status


def test_real_core_is_deterministic_rng_isolated_and_honestly_scoped():
    torch.manual_seed(98_765)
    rng_before = torch.random.get_rng_state().clone()

    first = run_conformance(profile="core")
    rng_after = torch.random.get_rng_state().clone()
    second = run_conformance(profile="core")

    first_payload = _json_payload(first)
    assert first == second
    assert first_payload == _json_payload(second)
    assert torch.equal(rng_before, rng_after)
    assert first.status == "pass"
    assert first.passed is True
    assert first.complete is True
    assert first.network == _network_receipt("core", attempted=False)
    assert first_payload.endswith("\n")
    assert not first_payload.endswith("\n\n")
    assert json.dumps(
        first.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n" == first_payload
    assert "official-weight runtime parity" in first.claim_boundary
    assert "performance" in first.claim_boundary
    assert "model quality" in first.claim_boundary
    for check in first.checks:
        assert check.claim_boundary
        if check.receipt is not None and "claim_boundary" in check.receipt:
            assert check.receipt["claim_boundary"] == check.claim_boundary
