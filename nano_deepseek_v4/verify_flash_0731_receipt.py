"""Reproduce the bundled DeepSeek-V4-Flash-0731 metadata receipt."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import re
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import date
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, Literal

from nano_deepseek_v4.checkpoint import inspect_deepseek_hub_checkpoint_namespace

PACKAGE_BOUNDARY = Path(__file__).resolve().parents[1]
RESOURCE_DIRECTORY = "_receipts"
RECEIPT_RESOURCE = "DeepSeek-V4-Flash-0731-metadata.json"
CONFIG_RESOURCE = "DeepSeek-V4-Flash-0731-config.json"
REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
FILES = (
    "config.json",
    "model.safetensors.index.json",
    "inference/config.json",
    "inference/model.py",
)
SCOPE = (
    "The result verifies the pinned config, checkpoint index, and selected "
    "DSpark shard headers. It does not verify weight payload integrity, "
    "full-snapshot presence, model quality, or native DSpark execution."
)
TOP_LEVEL_KEYS = {
    "schema_version",
    "cli_report_schema_version",
    "source",
    "source_revision",
    "validated_at",
    "metadata_probe_command",
    "official_source_sha256",
    "architecture",
    "checkpoint_index",
    "hub_metadata_probe",
    "dspark_namespace",
    "scope_boundary",
}
SCALE_SIDECAR_FIELD_TYPES = {
    "metadata_verified": "boolean",
    "one_to_one": "boolean",
    "dtype": "string",
    "fp8_weight_shape_rule": "string",
    "packed_fp4_expert_shape_rule": "string",
}
DTYPE_COUNT_FIELD_TYPES = {
    "BF16": "integer",
    "F32": "integer",
    "F8_E4M3": "integer",
    "F8_E8M0": "integer",
    "I8": "integer",
}
SECTION_FIELD_TYPES = {
    "architecture": {
        "backbone_layer_count": "integer",
        "raw_num_nextn_predict_layers": "integer",
        "dspark_stage_count": "integer",
        "dspark_layer_types": "string array",
        "dspark_block_size": "integer",
        "dspark_noise_token_id": "integer",
        "dspark_target_layer_ids": "integer array",
        "dspark_markov_rank": "integer",
        "runtime_load_supported": "boolean",
    },
    "checkpoint_index": {
        "indexed_tensor_count": "integer",
        "shard_count": "integer",
        "indexed_tensor_bytes": "integer",
    },
    "hub_metadata_probe": {
        "huggingface_hub_version": "string",
        "requested_revision": "string",
        "resolved_revision": "string",
        "metadata_only": "boolean",
        "verification_scope": "string",
        "metadata_document_bytes": "integer",
        "inspected_shard_count": "integer",
        "inspected_shard_files": "string array",
        "cached_safetensors_file_count": "integer",
        "snapshot_preflight_complete": "boolean",
        "payload_integrity_verified": "boolean",
    },
    "dspark_namespace": {
        "checkpoint_family": "string",
        "namespace": "string",
        "namespace_kind": "string",
        "schema_compatible": "boolean",
        "verification_scope": "string",
        "snapshot_preflight_complete": "boolean",
        "scale_metadata_verified": "boolean",
        "payload_integrity_verified": "boolean",
        "runtime_load_supported": "boolean",
        "indexed_tensor_count": "integer",
        "inspected_tensor_count": "integer",
        "non_scale_tensor_count": "integer",
        "scale_tensor_count": "integer",
        "scaled_tensor_count": "integer",
        "quantized_tensor_count": "integer",
        "scale_sidecar_validation": "scale sidecar object",
        "stored_tensor_bytes": "integer",
        "logical_model_parameter_count": "integer",
        "non_parameter_routing_state_count": "integer",
        "dtype_counts": "dtype count object",
        "inventory_sha256": "string",
        "inventory_sha256_scope": "string",
        "missing_expected_key_count": "integer",
        "missing_key_count": "integer",
        "unexpected_key_count": "integer",
        "unrecognized_key_count": "integer",
        "shape_mismatch_count": "integer",
        "unchecked_shape_key_count": "integer",
        "error_count": "integer",
        "is_complete": "boolean",
    },
}
SECTION_KEYS = {
    section_name: set(field_types)
    for section_name, field_types in SECTION_FIELD_TYPES.items()
}
Downloader = Callable[..., str | Path]
Inspector = Callable[..., Any]
NetworkAttemptObserver = Callable[[], None]
ReceiptUnavailableReason = Literal[
    "missing_optional_dependency",
    "source_unavailable",
    "inspection_unavailable",
]
_NETWORK_ERRNOS = frozenset(
    code
    for name in (
        "ECONNABORTED",
        "ECONNREFUSED",
        "ECONNRESET",
        "EHOSTDOWN",
        "EHOSTUNREACH",
        "ENETDOWN",
        "ENETRESET",
        "ENETUNREACH",
        "ETIMEDOUT",
    )
    if isinstance((code := getattr(errno, name, None)), int)
)


class ReceiptVerificationError(RuntimeError):
    def __init__(self, errors: str | list[str]):
        self.errors = [errors] if isinstance(errors, str) else errors
        super().__init__("; ".join(self.errors))


class ReceiptUnavailableError(ReceiptVerificationError):
    """A receipt check that could not run, rather than one that found drift."""

    def __init__(
        self,
        errors: str | list[str],
        *,
        reason_code: ReceiptUnavailableReason,
    ):
        self.reason_code: ReceiptUnavailableReason = reason_code
        super().__init__(errors)


class ReceiptExecutionError(RuntimeError):
    """A local or harness failure that is neither receipt drift nor unavailability."""


def _exception_chain(exc: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return chain


def _is_missing_optional_dependency(exc: BaseException) -> bool:
    for item in _exception_chain(exc):
        if not isinstance(item, ModuleNotFoundError):
            continue
        name = getattr(item, "name", None)
        if name == "huggingface_hub" or (
            isinstance(name, str) and name.startswith("huggingface_hub.")
        ):
            return True
    return False


def _is_semantic_source_failure(exc: BaseException) -> bool:
    """Recognize source absence from exception types, never their text.

    Hugging Face has used both ``EntryNotFoundError`` and the more specific
    ``RemoteEntryNotFoundError`` across supported client versions. A local
    cache miss is explicitly excluded because it says nothing about the pinned
    source revision.
    """

    semantic_names = {
        "EntryNotFoundError",
        "RemoteEntryNotFoundError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
    }
    for item in _exception_chain(exc):
        names = {base.__name__ for base in type(item).__mro__}
        if "LocalEntryNotFoundError" in names:
            continue
        if names & semantic_names:
            return True
    return False


def _http_status_code(exc: BaseException) -> int | None:
    for item in _exception_chain(exc):
        try:
            response = getattr(item, "response", None)
            status_code = getattr(response, "status_code", None)
        except Exception:
            continue
        if type(status_code) is int:
            return status_code
    return None


def _is_transient_unavailability(exc: BaseException) -> bool:
    if _is_semantic_source_failure(exc):
        return False
    status_code = _http_status_code(exc)
    if status_code in {408, 425, 429} or (
        status_code is not None and 500 <= status_code <= 599
    ):
        return True
    for item in _exception_chain(exc):
        names = {base.__name__ for base in type(item).__mro__}
        qualified_names = {
            (base.__module__.split(".", 1)[0], base.__name__)
            for base in type(item).__mro__
        }
        if "LocalEntryNotFoundError" in names:
            return True
        if isinstance(item, (ConnectionError, TimeoutError)):
            return True
        if isinstance(item, OSError) and item.errno in _NETWORK_ERRNOS:
            return True
        if qualified_names & {
            ("httpx", "RequestError"),
            ("httpx", "TransportError"),
        }:
            return True
        if qualified_names & {
            ("httpcore", "NetworkError"),
            ("httpcore", "TimeoutException"),
        }:
            return True
        if qualified_names & {
            ("requests", "ConnectionError"),
            ("requests", "Timeout"),
        }:
            return True
        if qualified_names & {
            ("socket", "gaierror"),
            ("socket", "herror"),
        }:
            return True
    return False


def _is_json_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "integer":
        return _is_json_integer(value)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "string array":
        return isinstance(value, list) and all(isinstance(item, str) for item in value)
    if expected == "integer array":
        return isinstance(value, list) and all(_is_json_integer(item) for item in value)
    raise AssertionError(f"unknown JSON field type: {expected}")


def _object_type_errors(
    value: Any,
    field_types: Mapping[str, str],
    path: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{path} must be a JSON object"]
    errors: list[str] = []
    if set(value) != set(field_types):
        errors.append(f"{path} field set drift")
    for field, expected in field_types.items():
        if field not in value:
            continue
        field_path = f"{path}.{field}"
        if expected == "scale sidecar object":
            errors.extend(
                _object_type_errors(
                    value[field],
                    SCALE_SIDECAR_FIELD_TYPES,
                    field_path,
                )
            )
        elif expected == "dtype count object":
            errors.extend(
                _object_type_errors(
                    value[field],
                    DTYPE_COUNT_FIELD_TYPES,
                    field_path,
                )
            )
        elif not _matches_json_type(value[field], expected):
            errors.append(f"{field_path} must be a JSON {expected}")
    return errors


def _section_type_errors(values: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for section_name, field_types in SECTION_FIELD_TYPES.items():
        errors.extend(
            _object_type_errors(
                values.get(section_name),
                field_types,
                section_name,
            )
        )
    return errors


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        return False
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return parsed.isoformat() == value


def _resource_bytes(name: str) -> bytes:
    try:
        return (
            resource_files("nano_deepseek_v4")
            .joinpath(RESOURCE_DIRECTORY)
            .joinpath(name)
            .read_bytes()
        )
    except OSError as exc:
        raise ReceiptExecutionError(
            f"bundled resource read failed for {name}: {type(exc).__name__}"
        ) from exc


def _input_bytes(path: Path | None, resource_name: str, label: str) -> bytes:
    if path is None:
        return _resource_bytes(resource_name)
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise ReceiptExecutionError(
            f"{label} read failed: {type(exc).__name__}"
        ) from exc


def _load(path: Path | None) -> dict[str, Any]:
    try:
        value = json.loads(_input_bytes(path, RECEIPT_RESOURCE, "receipt"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptVerificationError(f"cannot read receipt: {exc}") from exc
    if not isinstance(value, dict):
        raise ReceiptVerificationError("receipt must be a JSON object")
    return value


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReceiptExecutionError(
            f"source hash execution failed: {type(exc).__name__}"
        ) from exc
    return digest.hexdigest()


def _inputs(receipt: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    revision = receipt.get("source_revision")
    errors = []
    if set(receipt) != TOP_LEVEL_KEYS:
        errors.append("receipt top-level field set drift")
    schema_version = receipt.get("schema_version")
    cli_report_schema_version = receipt.get("cli_report_schema_version")
    if (
        not _is_json_integer(schema_version)
        or schema_version != 1
        or not _is_json_integer(cli_report_schema_version)
        or cli_report_schema_version != 3
    ):
        errors.append("receipt schema drift")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        errors.append("source_revision must be an immutable lowercase commit SHA")
        revision = ""
    if receipt.get("source") != f"https://huggingface.co/{REPO}":
        errors.append("source repository drift")
    if receipt.get("scope_boundary") != SCOPE:
        errors.append("scope boundary drift")
    if not _is_iso_date(receipt.get("validated_at")):
        errors.append("validated_at must be an ISO date")
    expected_command = (
        f"nano-deepseek-v4-inspect --hf-repo {REPO} --revision {revision} "
        "--namespace mtp --hf-cache-dir /path/to/empty-cache --json"
    )
    if receipt.get("metadata_probe_command") != expected_command:
        errors.append("metadata probe command drift")
    hashes = receipt.get("official_source_sha256")
    if not isinstance(hashes, Mapping):
        raise ReceiptVerificationError("official_source_sha256 must be an object")
    if set(hashes) != set(FILES):
        errors.append("official source file set drift")
    if any(
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in hashes.values()
    ):
        errors.append("invalid official source SHA-256")
    errors.extend(_section_type_errors(receipt))
    hub = receipt.get("hub_metadata_probe")
    if isinstance(hub, Mapping) and (
        not isinstance(hub.get("huggingface_hub_version"), str)
        or not hub["huggingface_hub_version"].strip()
    ):
        errors.append("receipt huggingface_hub_version must be a non-empty string")
    if errors:
        raise ReceiptVerificationError(errors)
    return revision, hashes


def _cache(path: Path) -> Path:
    try:
        cache = path.resolve()
    except OSError as exc:
        raise ReceiptExecutionError(
            f"dedicated cache setup failed: {type(exc).__name__}"
        ) from exc
    try:
        cache.relative_to(PACKAGE_BOUNDARY.resolve())
    except ValueError:
        pass
    else:
        raise ReceiptExecutionError(
            "dedicated cache must be outside the package source or installation"
        )
    try:
        if cache.exists() and (not cache.is_dir() or any(cache.iterdir())):
            raise ReceiptExecutionError("dedicated cache must be empty")
        cache.mkdir(parents=True, exist_ok=True)
    except ReceiptExecutionError:
        raise
    except OSError as exc:
        raise ReceiptExecutionError(
            f"dedicated cache setup failed: {type(exc).__name__}"
        ) from exc
    return cache


def _download(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    cache_dir: Path,
    on_network_attempt: NetworkAttemptObserver | None = None,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ReceiptUnavailableError(
            'install `nano-deepseek-v4[official]`',
            reason_code="missing_optional_dependency",
        ) from exc
    if on_network_attempt is not None:
        on_network_attempt()
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
            cache_dir=str(cache_dir),
        )
    )


def _inspect(
    *,
    repo_id: str,
    revision: str,
    namespace: str,
    cache_dir: Path,
    on_network_attempt: NetworkAttemptObserver | None = None,
) -> Any:
    if on_network_attempt is not None:
        on_network_attempt()
    return inspect_deepseek_hub_checkpoint_namespace(
        repo_id,
        revision,
        namespace,
        cache_dir=cache_dir,
    )


def _sections(report: Any, weights: int) -> dict[str, Any]:
    config, namespace = report.config, report.namespace
    aliases = {
        "backbone_layer_count": "num_hidden_layers",
        "raw_num_nextn_predict_layers": "num_nextn_predict_layers",
        "indexed_tensor_count": "total_indexed_tensor_count",
        "shard_count": "total_shard_count",
        "indexed_tensor_bytes": "index_total_size_bytes",
    }
    architecture = {
        key: getattr(config, aliases.get(key, key))
        for key in SECTION_KEYS["architecture"]
        if key != "runtime_load_supported"
    }
    architecture["runtime_load_supported"] = namespace.runtime_load_supported
    architecture["dspark_layer_types"] = list(architecture["dspark_layer_types"] or [])
    architecture["dspark_target_layer_ids"] = list(architecture["dspark_target_layer_ids"])
    checkpoint = {
        key: getattr(report, aliases[key])
        for key in SECTION_KEYS["checkpoint_index"]
    }
    computed_hub_keys = {
        "huggingface_hub_version",
        "cached_safetensors_file_count",
        "verification_scope",
        "snapshot_preflight_complete",
        "payload_integrity_verified",
    }
    hub = {
        key: getattr(report, key)
        for key in SECTION_KEYS["hub_metadata_probe"]
        if key not in computed_hub_keys
    }
    hub.update(
        huggingface_hub_version=report.huggingface_hub_version,
        cached_safetensors_file_count=weights,
        verification_scope=namespace.verification_scope,
        snapshot_preflight_complete=namespace.snapshot_preflight_complete,
        payload_integrity_verified=namespace.payload_integrity_verified,
    )
    counts = {
        "missing_expected_key_count": "missing_expected_keys",
        "missing_key_count": "missing_keys_in_shards",
        "unexpected_key_count": "unexpected_keys_in_shards",
        "unrecognized_key_count": "unrecognized_keys",
        "shape_mismatch_count": "shape_mismatches",
        "unchecked_shape_key_count": "unchecked_shape_keys",
        "error_count": "errors",
    }
    computed_dspark_keys = {
        *counts,
        "inventory_sha256_scope",
        "scale_sidecar_validation",
    }
    dspark = {
        key: getattr(namespace, key)
        for key in SECTION_KEYS["dspark_namespace"]
        if key not in computed_dspark_keys
    }
    for key, attribute in counts.items():
        collection = getattr(namespace, attribute)
        if not isinstance(collection, list):
            raise TypeError(f"namespace.{attribute} must be a list")
        dspark[key] = len(collection)
    dspark["inventory_sha256_scope"] = (
        "canonical sorted JSON mapping DSpark namespace keys to safetensors dtype and shape"
    )
    dspark["scale_sidecar_validation"] = {
        "metadata_verified": namespace.scale_metadata_verified,
        "one_to_one": namespace.scale_metadata_verified
        and namespace.scale_tensor_count == namespace.scaled_tensor_count,
        "dtype": "F8_E8M0",
        "fp8_weight_shape_rule": "ceil(out/128) x ceil(in/128)",
        "packed_fp4_expert_shape_rule": "out x ceil(logical_in/32)",
    }
    return {
        "architecture": architecture,
        "checkpoint_index": checkpoint,
        "hub_metadata_probe": hub,
        "dspark_namespace": dspark,
    }


def verify_flash_0731_receipt(
    receipt_path: Path | None = None,
    config_path: Path | None = None,
    *,
    cache_dir: Path,
    downloader: Downloader | None = None,
    inspector: Inspector | None = None,
    on_network_attempt: NetworkAttemptObserver | None = None,
) -> dict[str, Any]:
    receipt = _load(receipt_path)
    revision, hashes = _inputs(receipt)
    config_bytes = _input_bytes(config_path, CONFIG_RESOURCE, "config")
    if _hash_bytes(config_bytes) != hashes["config.json"]:
        raise ReceiptVerificationError("config SHA-256 drift")
    cache = _cache(cache_dir)
    actual: dict[str, str] = {}
    for filename in FILES:
        downloaded: str | Path
        try:
            if downloader is None:
                downloaded = _download(
                    repo_id=REPO,
                    revision=revision,
                    filename=filename,
                    cache_dir=cache,
                    on_network_attempt=on_network_attempt,
                )
            else:
                if on_network_attempt is not None:
                    on_network_attempt()
                downloaded = downloader(
                    repo_id=REPO,
                    revision=revision,
                    filename=filename,
                    cache_dir=cache,
                )
        except ReceiptUnavailableError as exc:
            raise ReceiptUnavailableError(
                f"source download unavailable for {filename}",
                reason_code=exc.reason_code,
            ) from exc
        except ReceiptVerificationError as exc:
            raise ReceiptVerificationError(
                f"source download verification failed for {filename}"
            ) from exc
        except ReceiptExecutionError:
            raise
        except Exception as exc:
            if _is_missing_optional_dependency(exc):
                raise ReceiptUnavailableError(
                    f"source download dependency unavailable for {filename}",
                    reason_code="missing_optional_dependency",
                ) from exc
            if _is_semantic_source_failure(exc):
                raise ReceiptVerificationError(
                    f"pinned source is absent for {filename}"
                ) from exc
            if _is_transient_unavailability(exc):
                raise ReceiptUnavailableError(
                    f"source download unavailable for {filename}",
                    reason_code="source_unavailable",
                ) from exc
            raise ReceiptExecutionError(
                f"source download execution failed for {filename}: {type(exc).__name__}"
            ) from exc
        try:
            path = Path(downloaded).resolve(strict=True)
        except Exception as exc:
            raise ReceiptExecutionError(
                f"downloaded source path validation failed for {filename}: "
                f"{type(exc).__name__}"
            ) from exc
        try:
            path.relative_to(cache)
        except ValueError as exc:
            raise ReceiptExecutionError(
                f"downloaded source escaped the dedicated cache for {filename}"
            ) from exc
        actual[filename] = _hash(path)
    mismatched = [name for name in FILES if actual.get(name) != hashes.get(name)]
    if mismatched:
        raise ReceiptVerificationError(f"official source SHA-256 drift: {mismatched}")
    try:
        if inspector is None:
            report = _inspect(
                repo_id=REPO,
                revision=revision,
                namespace="mtp",
                cache_dir=cache,
                on_network_attempt=on_network_attempt,
            )
        else:
            if on_network_attempt is not None:
                on_network_attempt()
            report = inspector(
                repo_id=REPO,
                revision=revision,
                namespace="mtp",
                cache_dir=cache,
            )
    except ReceiptUnavailableError as exc:
        raise ReceiptUnavailableError(
            "selected-header inspection unavailable",
            reason_code=exc.reason_code,
        ) from exc
    except ReceiptVerificationError as exc:
        raise ReceiptVerificationError("selected-header inspection verification failed") from exc
    except ReceiptExecutionError:
        raise
    except Exception as exc:
        if _is_missing_optional_dependency(exc):
            raise ReceiptUnavailableError(
                "selected-header inspection dependency unavailable",
                reason_code="missing_optional_dependency",
            ) from exc
        if _is_semantic_source_failure(exc):
            raise ReceiptVerificationError(
                "selected-header inspection found pinned source absence"
            ) from exc
        if inspector is None and isinstance(exc, ValueError):
            raise ReceiptVerificationError(
                "selected-header inspection found malformed pinned metadata"
            ) from exc
        if _is_transient_unavailability(exc):
            raise ReceiptUnavailableError(
                "selected-header inspection unavailable",
                reason_code="inspection_unavailable",
            ) from exc
        raise ReceiptExecutionError(
            f"selected-header inspection execution failed: {type(exc).__name__}"
        ) from exc
    weight_files = [path for path in cache.rglob("*.safetensors") if path.is_file()]
    errors: list[str] = []
    try:
        report_repo_id = report.repo_id
        requested_revision = report.requested_revision
        resolved_revision = report.resolved_revision
        config_sha256 = report.config_sha256
        index_sha256 = report.index_sha256
        report_is_complete = report.is_complete
        namespace_is_complete = report.namespace.is_complete
        metadata_only = report.metadata_only
        payload_integrity_verified = report.namespace.payload_integrity_verified
        runtime_load_supported = report.namespace.runtime_load_supported
        observed_hub_version = report.huggingface_hub_version
        sections = _sections(report, len(weight_files))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ReceiptVerificationError(
            f"malformed inspection report: {type(exc).__name__}"
        ) from exc
    malformed_report = _section_type_errors(sections)
    if not isinstance(report_is_complete, bool):
        malformed_report.append("report.is_complete must be a JSON boolean")
    if malformed_report:
        raise ReceiptVerificationError(
            [f"malformed inspection report: {error}" for error in malformed_report]
        )
    if not isinstance(observed_hub_version, str) or not observed_hub_version.strip():
        raise ReceiptVerificationError(
            "observed huggingface_hub_version must be a non-empty string"
        )
    live = (
        report_repo_id == REPO
        and requested_revision == revision == resolved_revision
        and config_sha256 == hashes["config.json"]
        and index_sha256 == hashes["model.safetensors.index.json"]
        and report_is_complete
        and namespace_is_complete
        and metadata_only
        and not payload_integrity_verified
        and not runtime_load_supported
    )
    if not live:
        errors.append("live inspection provenance or scope drift")
    if weight_files:
        errors.append(
            "metadata probe cached safetensors files: "
            f"{[path.name for path in weight_files]}"
        )
    for name, value in sections.items():
        expected = dict(receipt[name])
        observed = dict(value)
        if name == "hub_metadata_probe":
            expected.pop("huggingface_hub_version")
            observed.pop("huggingface_hub_version")
        if observed != expected:
            errors.append(f"{name} drift")
    if errors:
        raise ReceiptVerificationError(errors)
    return {
        "schema_version": 1,
        "status": "pass",
        "repo_id": REPO,
        "revision": revision,
        "official_source_sha256": actual,
        "resolved_revision": resolved_revision,
        "observed_huggingface_hub_version": observed_hub_version,
        "receipt_huggingface_hub_version": receipt["hub_metadata_probe"][
            "huggingface_hub_version"
        ],
        "verification_scope": sections["dspark_namespace"]["verification_scope"],
        "inspected_shard_count": sections["hub_metadata_probe"]["inspected_shard_count"],
        "cached_safetensors_file_count": 0,
        "scope_boundary": SCOPE,
    }


def main(
    argv: list[str] | None = None,
    *,
    downloader: Downloader | None = None,
    inspector: Inspector | None = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="custom receipt JSON (must be supplied with --config)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="custom config JSON (must be supplied with --receipt)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if (args.receipt is None) != (args.config is None):
        parser.error("--receipt and --config must be supplied together")
    try:
        with tempfile.TemporaryDirectory(prefix="nano-dsv4-0731-") as temp:
            result = verify_flash_0731_receipt(
                args.receipt,
                args.config,
                cache_dir=Path(temp),
                downloader=downloader,
                inspector=inspector,
            )
    except ReceiptUnavailableError as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "incomplete",
                        "reason_code": exc.reason_code,
                    }
                )
            )
        else:
            print(
                f"receipt verification incomplete: {exc.reason_code}",
                file=sys.stderr,
            )
        return 3
    except ReceiptVerificationError as exc:
        if args.json:
            print(json.dumps({"schema_version": 1, "status": "fail", "errors": exc.errors}))
        else:
            print(f"receipt verification failed: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "error",
                        "error_type": type(exc).__name__,
                    }
                )
            )
        else:
            print(f"receipt verification error: {type(exc).__name__}", file=sys.stderr)
        return 4
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"verified {REPO}@{result['revision']}\n"
            f"4 source hashes; {result['inspected_shard_count']} selected shard headers; "
            "0 cached .safetensors files\n"
            f"scope: {SCOPE}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
