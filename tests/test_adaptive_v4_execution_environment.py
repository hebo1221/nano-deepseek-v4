from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import adaptive_v4_execution_environment as environment  # noqa: E402


def _environment() -> dict[str, Any]:
    return {
        "schema_version": environment.SCHEMA_VERSION,
        "python_implementation": "CPython",
        "python_version": "3.12.0",
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": "2.7.0+cu128",
        "cuda_runtime_version": "12.8",
        "cuda_driver_version": "570.00",
        "cuda_visible_devices": "0,1",
        "platform_system": "Linux",
        "platform_release": "test-kernel",
        "platform_machine": "x86_64",
        "platform_string": "Linux-test-x86_64",
        "current_device_index": 1,
        "visible_device_count": 2,
        "visible_devices": [
            {
                "logical_index": 0,
                "name": "GPU-A",
                "uuid": "GPU-a",
                "pci_bus_id": "0000:01:00.0",
                "compute_capability": [8, 0],
                "total_memory_bytes": 40 * 1024**3,
            },
            {
                "logical_index": 1,
                "name": "GPU-B",
                "uuid": "GPU-b",
                "pci_bus_id": "0000:02:00.0",
                "compute_capability": [9, 0],
                "total_memory_bytes": 80 * 1024**3,
            },
        ],
    }


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_execution_environment_schema_is_exact(mutation: str) -> None:
    payload = _environment()
    if mutation == "missing":
        payload.pop("platform_string")
    else:
        payload["unregistered"] = True

    with pytest.raises(ValueError, match="schema drifted"):
        environment.validate_execution_environment(payload)


def test_selected_device_class_uses_current_device_not_first_visible_gpu() -> None:
    payload = _environment()
    assert environment.selected_device_context(payload) == {
        "device_spec": "cuda:1",
        "logical_device_index": 1,
    }
    assert environment.selected_device_class(payload) == {
        "name": "GPU-B",
        "compute_capability": [9, 0],
        "total_memory_bytes": 80 * 1024**3,
    }
    assert environment.selected_device_routing_identity(payload) == {
        "identity_type": "uuid",
        "identity": "GPU-b",
    }

    no_uuid = copy.deepcopy(payload)
    no_uuid["visible_devices"][1]["uuid"] = ""
    assert environment.selected_device_routing_identity(no_uuid) == {
        "identity_type": "pci_bus_id",
        "identity": "0000:02:00.0",
    }


def test_controller_projection_ignores_routing_but_not_selected_device_class() -> None:
    baseline = _environment()
    routing_only = copy.deepcopy(baseline)
    routing_only["cuda_visible_devices"] = "7"
    routing_only["current_device_index"] = 0
    routing_only["visible_device_count"] = 1
    selected = copy.deepcopy(baseline["visible_devices"][1])
    selected["logical_index"] = 0
    selected["uuid"] = "GPU-rerouted"
    selected["pci_bus_id"] = "0000:07:00.0"
    routing_only["visible_devices"] = [selected]

    assert environment.controller_compatible_environment_projection(baseline) == (
        environment.controller_compatible_environment_projection(routing_only)
    )

    wrong_selection = copy.deepcopy(baseline)
    wrong_selection["current_device_index"] = 0
    assert environment.controller_compatible_environment_projection(baseline) != (
        environment.controller_compatible_environment_projection(wrong_selection)
    )


def test_explicit_nonzero_cuda_device_is_activated_before_environment_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _environment()
    activated: list[str] = []
    monkeypatch.setattr(environment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(environment.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        environment.torch.cuda,
        "set_device",
        lambda device: activated.append(str(device)),
    )
    monkeypatch.setattr(
        environment,
        "capture_execution_environment",
        lambda: copy.deepcopy(payload),
    )

    device, captured = environment.activate_explicit_cuda_device(
        "cuda:1",
        expected_routing_identity={"identity_type": "uuid", "identity": "GPU-b"},
    )

    assert str(device) == "cuda:1"
    assert activated == ["cuda:1"]
    assert captured == payload


def test_cuda_activation_rejects_device_that_does_not_match_guard_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _environment()
    monkeypatch.setattr(environment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(environment.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(environment.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        environment,
        "capture_execution_environment",
        lambda: copy.deepcopy(payload),
    )

    with pytest.raises(ValueError, match="physical-device guard identity"):
        environment.activate_explicit_cuda_device(
            "cuda:1",
            expected_routing_identity={"identity_type": "uuid", "identity": "GPU-a"},
        )


@pytest.mark.parametrize("device_spec", ["cuda", "cpu", "cuda:-1"])
def test_cuda_activation_rejects_ambiguous_or_non_cuda_routes(device_spec: str) -> None:
    with pytest.raises(ValueError, match="explicit logical CUDA device|invalid"):
        environment.activate_explicit_cuda_device(device_spec)
