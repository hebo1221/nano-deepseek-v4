from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch

SCHEMA_VERSION = 1

EXECUTION_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema_version",
        "python_implementation",
        "python_version",
        "python_executable",
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "cuda_visible_devices",
        "platform_system",
        "platform_release",
        "platform_machine",
        "platform_string",
        "current_device_index",
        "visible_device_count",
        "visible_devices",
    }
)
VISIBLE_DEVICE_FIELDS = frozenset(
    {
        "logical_index",
        "name",
        "uuid",
        "pci_bus_id",
        "compute_capability",
        "total_memory_bytes",
    }
)
SELECTED_DEVICE_CLASS_FIELDS = frozenset({"name", "compute_capability", "total_memory_bytes"})
SELECTED_DEVICE_ROUTING_IDENTITY_FIELDS = frozenset({"identity_type", "identity"})
DEVICE_CONTEXT_FIELDS = frozenset({"device_spec", "logical_device_index"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def query_cuda_driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("CUDA driver version could not be queried safely.") from error
    _require(result.returncode == 0, "CUDA driver version query failed.")
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    _require(len(versions) == 1, "CUDA driver version query was empty or inconsistent.")
    return versions.pop()


def validate_execution_environment(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        set(payload) == EXECUTION_ENVIRONMENT_FIELDS,
        "CUDA execution environment schema drifted.",
    )
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        "CUDA execution environment version drifted.",
    )
    for field in (
        "python_implementation",
        "python_version",
        "python_executable",
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "platform_system",
        "platform_release",
        "platform_machine",
        "platform_string",
    ):
        _require(
            isinstance(payload.get(field), str) and bool(cast(str, payload[field]).strip()),
            f"CUDA execution environment {field} is missing.",
        )
    executable = Path(cast(str, payload["python_executable"]))
    _require(
        executable == Path(os.path.abspath(executable)),
        "CUDA execution environment Python executable is not absolute.",
    )
    cuda_visible_devices = payload.get("cuda_visible_devices")
    _require(
        cuda_visible_devices is None or isinstance(cuda_visible_devices, str),
        "CUDA_VISIBLE_DEVICES provenance is invalid.",
    )
    count = payload.get("visible_device_count")
    current = payload.get("current_device_index")
    devices = payload.get("visible_devices")
    _require(
        type(count) is int
        and count > 0
        and type(current) is int
        and 0 <= current < count
        and isinstance(devices, list)
        and len(devices) == count,
        "Visible CUDA device inventory is invalid.",
    )
    validated_devices: list[dict[str, Any]] = []
    routing_identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(cast(list[Any], devices)):
        _require(isinstance(raw, Mapping), "Visible CUDA device record is invalid.")
        device = cast(Mapping[str, Any], raw)
        _require(set(device) == VISIBLE_DEVICE_FIELDS, "Visible CUDA device schema drifted.")
        capability = device.get("compute_capability")
        uuid = device.get("uuid")
        pci_bus_id = device.get("pci_bus_id")
        _require(
            device.get("logical_index") == index
            and isinstance(device.get("name"), str)
            and bool(cast(str, device["name"]).strip())
            and isinstance(uuid, str)
            and isinstance(pci_bus_id, str)
            and (bool(uuid) or bool(pci_bus_id))
            and isinstance(capability, list)
            and len(capability) == 2
            and all(type(value) is int and value >= 0 for value in capability)
            and type(device.get("total_memory_bytes")) is int
            and cast(int, device["total_memory_bytes"]) > 0,
            "Visible CUDA device identity is invalid.",
        )
        identity = (cast(str, uuid), cast(str, pci_bus_id))
        _require(identity not in routing_identities, "Visible CUDA device identity is repeated.")
        routing_identities.add(identity)
        validated_devices.append(dict(device))
    return {**dict(payload), "visible_devices": validated_devices}


def capture_execution_environment() -> dict[str, Any]:
    try:
        _require(torch.cuda.is_available(), "Execution requires available CUDA.")
        device_count = torch.cuda.device_count()
        _require(device_count > 0, "Execution has no visible CUDA devices.")
        current_device = torch.cuda.current_device()
        devices: list[dict[str, Any]] = []
        for logical_index in range(device_count):
            properties = torch.cuda.get_device_properties(logical_index)
            raw_uuid = getattr(properties, "uuid", None)
            raw_pci_bus_id = getattr(properties, "pci_bus_id", None)
            devices.append(
                {
                    "logical_index": logical_index,
                    "name": str(properties.name),
                    "uuid": "" if raw_uuid is None else str(raw_uuid),
                    "pci_bus_id": "" if raw_pci_bus_id is None else str(raw_pci_bus_id),
                    "compute_capability": [int(properties.major), int(properties.minor)],
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
        cuda_runtime = torch.version.cuda
        _require(
            isinstance(cuda_runtime, str) and bool(cuda_runtime),
            "PyTorch CUDA runtime version is unavailable.",
        )
        return validate_execution_environment(
            {
                "schema_version": SCHEMA_VERSION,
                "python_implementation": platform.python_implementation(),
                "python_version": platform.python_version(),
                "python_executable": str(Path(sys.executable).resolve(strict=True)),
                "torch_version": str(torch.__version__),
                "cuda_runtime_version": cuda_runtime,
                "cuda_driver_version": query_cuda_driver_version(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "platform_system": platform.system(),
                "platform_release": platform.release(),
                "platform_machine": platform.machine(),
                "platform_string": platform.platform(aliased=False, terse=False),
                "current_device_index": current_device,
                "visible_device_count": device_count,
                "visible_devices": devices,
            }
        )
    except ValueError:
        raise
    except (OSError, RuntimeError, TypeError, AttributeError) as error:
        raise ValueError("CUDA execution environment could not be captured safely.") from error


def assert_exact_execution_environment(expected: Mapping[str, Any]) -> None:
    frozen = validate_execution_environment(expected)
    current = capture_execution_environment()
    _require(current == frozen, "CUDA execution environment changed after freeze.")


def explicit_cuda_device_spec(payload: Mapping[str, Any]) -> str:
    """Return the unambiguous logical CUDA route selected by a frozen environment."""

    validated = validate_execution_environment(payload)
    return f"cuda:{validated['current_device_index']}"


def selected_device_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_execution_environment(payload)
    context = {
        "device_spec": explicit_cuda_device_spec(validated),
        "logical_device_index": validated["current_device_index"],
    }
    _require(set(context) == DEVICE_CONTEXT_FIELDS, "CUDA device context schema drifted.")
    return context


def validate_device_routing_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    _require(
        set(payload) == SELECTED_DEVICE_ROUTING_IDENTITY_FIELDS
        and payload.get("identity_type") in {"uuid", "pci_bus_id"}
        and isinstance(payload.get("identity"), str)
        and bool(cast(str, payload["identity"])),
        "CUDA device routing identity is invalid.",
    )
    return {
        "identity_type": cast(str, payload["identity_type"]),
        "identity": cast(str, payload["identity"]),
    }


def activate_explicit_cuda_device(
    device_spec: str,
    *,
    expected_routing_identity: Mapping[str, Any] | None = None,
) -> tuple[torch.device, dict[str, Any]]:
    """Activate an explicit logical CUDA device and attest the route after activation."""

    _require(isinstance(device_spec, str), "CUDA device specification is invalid.")
    try:
        device = torch.device(device_spec)
    except (RuntimeError, ValueError, TypeError) as error:
        raise ValueError("CUDA device specification is invalid.") from error
    _require(
        device.type == "cuda" and type(device.index) is int and device.index >= 0,
        "Execution requires an explicit logical CUDA device such as cuda:0.",
    )
    try:
        _require(torch.cuda.is_available(), "Execution requires available CUDA.")
        _require(
            device.index < torch.cuda.device_count(),
            "Explicit logical CUDA device is outside the visible inventory.",
        )
        torch.cuda.set_device(device)
    except ValueError:
        raise
    except (RuntimeError, TypeError) as error:
        raise ValueError("Explicit logical CUDA device could not be activated.") from error
    captured = capture_execution_environment()
    _require(
        captured["current_device_index"] == device.index,
        "Activated logical CUDA device does not match the captured execution route.",
    )
    _require(
        selected_device_context(captured)
        == {"device_spec": str(device), "logical_device_index": device.index},
        "Activated CUDA device context drifted after capture.",
    )
    if expected_routing_identity is not None:
        _require(
            selected_device_routing_identity(captured)
            == validate_device_routing_identity(expected_routing_identity),
            "Activated CUDA device does not match the physical-device guard identity.",
        )
    return device, captured


def selected_device_class(payload: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_execution_environment(payload)
    current = cast(int, validated["current_device_index"])
    device = cast(list[dict[str, Any]], validated["visible_devices"])[current]
    selected = {
        "name": device["name"],
        "compute_capability": list(cast(list[int], device["compute_capability"])),
        "total_memory_bytes": device["total_memory_bytes"],
    }
    _require(
        set(selected) == SELECTED_DEVICE_CLASS_FIELDS,
        "Selected CUDA device class schema drifted.",
    )
    return selected


def selected_device_routing_identity(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return the strongest stable physical-routing identity for the selected GPU."""

    validated = validate_execution_environment(payload)
    current = cast(int, validated["current_device_index"])
    device = cast(list[dict[str, Any]], validated["visible_devices"])[current]
    uuid = cast(str, device["uuid"])
    pci_bus_id = cast(str, device["pci_bus_id"])
    selected = validate_device_routing_identity(
        {
            "identity_type": "uuid" if uuid else "pci_bus_id",
            "identity": uuid if uuid else pci_bus_id,
        }
    )
    return selected


def controller_compatible_environment_projection(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve software/platform and the GPU actually selected, not routing topology."""

    validated = validate_execution_environment(payload)
    return {
        "schema_version": validated["schema_version"],
        "python_implementation": validated["python_implementation"],
        "python_version": validated["python_version"],
        "python_executable": validated["python_executable"],
        "torch_version": validated["torch_version"],
        "cuda_runtime_version": validated["cuda_runtime_version"],
        "cuda_driver_version": validated["cuda_driver_version"],
        "platform_system": validated["platform_system"],
        "platform_release": validated["platform_release"],
        "platform_machine": validated["platform_machine"],
        "platform_string": validated["platform_string"],
        "selected_device_class": selected_device_class(validated),
    }
