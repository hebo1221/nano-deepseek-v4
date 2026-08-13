from __future__ import annotations

import hashlib
import json
from importlib.resources import files as resource_files
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import nano_deepseek_v4.verify_flash_0731_receipt as verifier
from nano_deepseek_v4.config import DeepSeekV4Config
from nano_deepseek_v4.verify_flash_0731_receipt import (
    CONFIG_RESOURCE,
    RECEIPT_RESOURCE,
    ReceiptExecutionError,
    ReceiptUnavailableError,
    ReceiptVerificationError,
    main,
    verify_flash_0731_receipt,
)

REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
FILES = (
    "config.json",
    "model.safetensors.index.json",
    "inference/config.json",
    "inference/model.py",
)


def _bundled_bytes(name: str) -> bytes:
    return (
        resource_files("nano_deepseek_v4")
        .joinpath("_receipts")
        .joinpath(name)
        .read_bytes()
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes], dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(_bundled_bytes(RECEIPT_RESOURCE))
    sources = {
        "config.json": _bundled_bytes(CONFIG_RESOURCE),
        "model.safetensors.index.json": b'{"metadata":{"total_size":166878536440}}',
        "inference/config.json": b'{"fixture":"config"}',
        "inference/model.py": b"raise RuntimeError('downloaded source was executed')\n",
    }
    receipt["official_source_sha256"] = {
        name: hashlib.sha256(data).hexdigest() for name, data in sources.items()
    }
    receipt_path, config_path = tmp_path / "receipt.json", tmp_path / "config.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    config_path.write_bytes(sources["config.json"])
    return receipt_path, config_path, sources, receipt


def _report(receipt: dict[str, Any]) -> Any:
    revision = receipt["source_revision"]
    hashes = receipt["official_source_sha256"]
    checkpoint, hub, dspark = (
        receipt["checkpoint_index"],
        receipt["hub_metadata_probe"],
        receipt["dspark_namespace"],
    )
    values = dict(dspark)
    values.pop("scale_sidecar_validation")
    values.pop("inventory_sha256_scope")
    counts = {
        "missing_expected_key_count": "missing_expected_keys",
        "missing_key_count": "missing_keys_in_shards",
        "unexpected_key_count": "unexpected_keys_in_shards",
        "unrecognized_key_count": "unrecognized_keys",
        "shape_mismatch_count": "shape_mismatches",
        "unchecked_shape_key_count": "unchecked_shape_keys",
        "error_count": "errors",
    }
    for count, target in counts.items():
        values[target] = [target] * values.pop(count)
    values.update(key_prefix="mtp.", index_sha256=hashes["model.safetensors.index.json"])
    namespace = SimpleNamespace(**values)
    return SimpleNamespace(
        repo_id=REPO,
        requested_revision=revision,
        resolved_revision=revision,
        huggingface_hub_version=hub["huggingface_hub_version"],
        config_sha256=hashes["config.json"],
        index_sha256=hashes["model.safetensors.index.json"],
        total_indexed_tensor_count=checkpoint["indexed_tensor_count"],
        total_shard_count=checkpoint["shard_count"],
        index_total_size_bytes=checkpoint["indexed_tensor_bytes"],
        metadata_document_bytes=hub["metadata_document_bytes"],
        inspected_shard_count=hub["inspected_shard_count"],
        inspected_shard_files=hub["inspected_shard_files"],
        metadata_only=hub["metadata_only"],
        namespace=namespace,
        is_complete=dspark["is_complete"],
        config=DeepSeekV4Config.flash_0731(),
    )


def _fakes(
    sources: dict[str, bytes],
    report: Any,
    *,
    corrupt: str | None = None,
    cache_weight: bool = False,
) -> tuple[Any, Any, list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    downloads: list[tuple[str, str, str]] = []
    inspections: list[tuple[str, str, str]] = []

    def download(**kwargs: Any) -> Path:
        repo, revision, name, cache = (
            kwargs["repo_id"], kwargs["revision"], kwargs["filename"], kwargs["cache_dir"]
        )
        downloads.append((repo, revision, name))
        data = sources[name] + (b"drift" if name == corrupt else b"")
        target = cache / "downloads" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    def inspect(**kwargs: Any) -> Any:
        inspections.append((kwargs["repo_id"], kwargs["revision"], kwargs["namespace"]))
        if cache_weight:
            weight = kwargs["cache_dir"] / "nested" / "weight.safetensors"
            weight.parent.mkdir(parents=True, exist_ok=True)
            weight.write_bytes(b"forbidden")
        return report

    return download, inspect, downloads, inspections


def test_reproduces_receipt_without_weight_downloads(tmp_path: Path):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, inspect, downloads, inspections = _fakes(sources, _report(receipt))
    result = verify_flash_0731_receipt(
        receipt_path,
        config_path,
        cache_dir=tmp_path / "cache",
        downloader=download,
        inspector=inspect,
    )
    assert result["status"] == "pass"
    assert result["cached_safetensors_file_count"] == 0
    assert result["observed_huggingface_hub_version"] == receipt["hub_metadata_probe"][
        "huggingface_hub_version"
    ]
    assert [name for _, _, name in downloads] == list(FILES)
    assert all(not name.endswith(".safetensors") for _, _, name in downloads)
    assert inspections == [(REPO, receipt["source_revision"], "mtp")]


def test_network_observer_counts_opaque_downloader_and_inspector_calls(
    tmp_path: Path,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    base_download, base_inspect, downloads, inspections = _fakes(
        sources,
        _report(receipt),
    )
    events: list[tuple[str, str | None]] = []

    def observe() -> None:
        events.append(("attempt", None))

    def download(**kwargs: Any) -> Path:
        events.append(("download", kwargs["filename"]))
        return base_download(**kwargs)

    def inspect(**kwargs: Any) -> Any:
        events.append(("inspect", kwargs["namespace"]))
        return base_inspect(**kwargs)

    result = verify_flash_0731_receipt(
        receipt_path,
        config_path,
        cache_dir=tmp_path / "cache",
        downloader=download,
        inspector=inspect,
        on_network_attempt=observe,
    )

    assert result["status"] == "pass"
    assert events == [
        ("attempt", None),
        ("download", "config.json"),
        ("attempt", None),
        ("download", "model.safetensors.index.json"),
        ("attempt", None),
        ("download", "inference/config.json"),
        ("attempt", None),
        ("download", "inference/model.py"),
        ("attempt", None),
        ("inspect", "mtp"),
    ]
    assert len(downloads) == 4
    assert len(inspections) == 1


def test_default_inputs_are_the_bundled_receipt_and_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt = json.loads(_bundled_bytes(RECEIPT_RESOURCE))
    assert hashlib.sha256(_bundled_bytes(CONFIG_RESOURCE)).hexdigest() == receipt[
        "official_source_sha256"
    ]["config.json"]

    placeholders = {name: b"hash-only fixture" for name in FILES}
    report = _report(receipt)
    download, inspect, downloads, inspections = _fakes(placeholders, report)
    cache = tmp_path / "cache"

    def expected_download_hash(path: Path) -> str:
        name = path.resolve().relative_to((cache / "downloads").resolve()).as_posix()
        return receipt["official_source_sha256"][name]

    monkeypatch.setattr(verifier, "_hash", expected_download_hash)

    result = verify_flash_0731_receipt(
        cache_dir=cache,
        downloader=download,
        inspector=inspect,
    )

    assert result["status"] == "pass"
    assert [name for _, _, name in downloads] == list(FILES)
    assert inspections == [(REPO, receipt["source_revision"], "mtp")]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("local_config", "config SHA-256 drift"),
        ("schema", "receipt schema drift"),
        ("revision", "immutable lowercase"),
        ("scope", "scope boundary drift"),
        ("source", "official source SHA-256 drift"),
        ("report", "dspark_namespace drift"),
        ("weight", "cached safetensors files"),
    ],
)
def test_rejects_drift_and_scope_expansion(tmp_path: Path, case: str, message: str):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    if case == "local_config":
        config_path.write_bytes(config_path.read_bytes() + b"drift")
    elif case == "schema":
        receipt["schema_version"] = 2
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif case == "revision":
        receipt["source_revision"] = "main"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif case == "scope":
        receipt["scope_boundary"] = "broader than the implemented verifier"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    elif case == "report":
        report.namespace.inventory_sha256 = "0" * 64
    download, inspect, downloads, _ = _fakes(
        sources,
        report,
        corrupt="inference/model.py" if case == "source" else None,
        cache_weight=case == "weight",
    )
    with pytest.raises(ReceiptVerificationError, match=message):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )
    if case in {"local_config", "schema", "revision", "scope"}:
        assert downloads == []


def test_hub_client_version_is_observed_provenance_not_semantic_drift(
    tmp_path: Path,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    report.huggingface_hub_version = "999.0.0"
    download, inspect, _, _ = _fakes(sources, report)

    result = verify_flash_0731_receipt(
        receipt_path,
        config_path,
        cache_dir=tmp_path / "cache",
        downloader=download,
        inspector=inspect,
    )

    assert result["observed_huggingface_hub_version"] == "999.0.0"
    assert result["receipt_huggingface_hub_version"] == receipt[
        "hub_metadata_probe"
    ]["huggingface_hub_version"]


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        ("architecture", lambda value: value.pop("dspark_markov_rank")),
        ("checkpoint_index", lambda value: value.pop("indexed_tensor_bytes")),
        ("hub_metadata_probe", lambda value: value.pop("metadata_document_bytes")),
        ("dspark_namespace", lambda value: value.pop("inventory_sha256")),
        ("architecture", lambda value: value.__setitem__("hidden_size", 4096)),
        ("checkpoint_index", lambda value: value.__setitem__("unknown", 1)),
        ("hub_metadata_probe", lambda value: value.__setitem__("repo_id", REPO)),
        ("dspark_namespace", lambda value: value.__setitem__("key_prefix", "mtp.")),
    ],
)
def test_rejects_malformed_receipt_section_shape_before_download(
    tmp_path: Path,
    section: str,
    mutate,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    mutate(receipt[section])
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    download, inspect, downloads, _ = _fakes(sources, report)

    with pytest.raises(ReceiptVerificationError, match=f"{section} field set drift"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )

    assert downloads == []


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_rejects_top_level_field_set_drift_before_download(
    tmp_path: Path,
    case: str,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    if case == "missing":
        receipt.pop("validated_at")
    else:
        receipt["unversioned_extension"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    download, inspect, downloads, _ = _fakes(sources, _report(receipt))

    with pytest.raises(ReceiptVerificationError, match="top-level field set drift"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )

    assert downloads == []


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        pytest.param(
            ("schema_version",),
            True,
            "receipt schema drift",
            id="schema-bool-is-not-int",
        ),
        pytest.param(
            ("cli_report_schema_version",),
            3.0,
            "receipt schema drift",
            id="schema-float-is-not-int",
        ),
        pytest.param(
            ("architecture", "backbone_layer_count"),
            43.0,
            "must be a JSON integer",
            id="architecture-float-is-not-int",
        ),
        pytest.param(
            ("architecture", "runtime_load_supported"),
            0,
            "must be a JSON boolean",
            id="architecture-int-is-not-bool",
        ),
        pytest.param(
            ("architecture", "dspark_target_layer_ids"),
            [40.0, 41, 42],
            "must be a JSON integer array",
            id="integer-array-element",
        ),
        pytest.param(
            ("hub_metadata_probe", "metadata_only"),
            1,
            "must be a JSON boolean",
            id="hub-int-is-not-bool",
        ),
        pytest.param(
            ("hub_metadata_probe", "metadata_document_bytes"),
            True,
            "must be a JSON integer",
            id="hub-bool-is-not-int",
        ),
        pytest.param(
            ("hub_metadata_probe", "inspected_shard_files"),
            [48],
            "must be a JSON string array",
            id="string-array-element",
        ),
        pytest.param(
            ("dspark_namespace", "is_complete"),
            1,
            "must be a JSON boolean",
            id="namespace-int-is-not-bool",
        ),
        pytest.param(
            ("dspark_namespace", "scale_sidecar_validation", "one_to_one"),
            1,
            "must be a JSON boolean",
            id="nested-int-is-not-bool",
        ),
        pytest.param(
            ("dspark_namespace", "dtype_counts", "BF16"),
            20.0,
            "must be a JSON integer",
            id="mapping-float-is-not-int",
        ),
    ],
)
def test_rejects_json_type_confusion_before_download(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    message: str,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    target: Any = receipt
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = replacement
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    download, inspect, downloads, inspections = _fakes(sources, report)

    with pytest.raises(ReceiptVerificationError, match=message):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )

    assert downloads == []
    assert inspections == []


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_rejects_nested_receipt_field_set_drift_before_download(
    tmp_path: Path,
    case: str,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    sidecar = receipt["dspark_namespace"]["scale_sidecar_validation"]
    if case == "missing":
        sidecar.pop("one_to_one")
    else:
        sidecar["unchecked_extension"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    download, inspect, downloads, inspections = _fakes(sources, report)

    with pytest.raises(ReceiptVerificationError, match="scale_sidecar_validation field set drift"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )

    assert downloads == []
    assert inspections == []


@pytest.mark.parametrize("validated_at", ["2026-02-31", "2026-13-01", "2026-2-03"])
def test_rejects_invalid_calendar_date_before_download(
    tmp_path: Path,
    validated_at: str,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    receipt["validated_at"] = validated_at
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    download, inspect, downloads, inspections = _fakes(sources, report)

    with pytest.raises(ReceiptVerificationError, match="validated_at must be an ISO date"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )

    assert downloads == []
    assert inspections == []


@pytest.mark.parametrize("version", ["", None])
def test_rejects_invalid_receipt_hub_version_before_download(
    tmp_path: Path,
    version: Any,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    receipt["hub_metadata_probe"]["huggingface_hub_version"] = version
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    download, inspect, downloads, _ = _fakes(sources, _report(receipt))

    with pytest.raises(ReceiptVerificationError, match="non-empty string"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )

    assert downloads == []


def test_hub_version_is_the_only_non_semantic_hub_field(tmp_path: Path):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    report.huggingface_hub_version = "999.0.0"
    report.metadata_document_bytes += 1
    download, inspect, _, _ = _fakes(sources, report)

    with pytest.raises(ReceiptVerificationError, match="hub_metadata_probe drift"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )


@pytest.mark.parametrize(
    "case",
    [
        "report_complete_int",
        "metadata_only_int",
        "namespace_complete_int",
        "runtime_supported_int",
        "checkpoint_count_float",
        "dtype_count_float",
    ],
)
def test_rejects_live_report_type_confusion(
    tmp_path: Path,
    case: str,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    if case == "report_complete_int":
        report.is_complete = 1
    elif case == "metadata_only_int":
        report.metadata_only = 1
    elif case == "namespace_complete_int":
        report.namespace.is_complete = 1
    elif case == "runtime_supported_int":
        report.namespace.runtime_load_supported = 0
    elif case == "checkpoint_count_float":
        report.total_indexed_tensor_count = float(report.total_indexed_tensor_count)
    else:
        report.namespace.dtype_counts = dict(report.namespace.dtype_counts)
        report.namespace.dtype_counts["BF16"] = float(
            report.namespace.dtype_counts["BF16"]
        )
    download, inspect, downloads, inspections = _fakes(sources, report)

    with pytest.raises(ReceiptVerificationError, match="malformed inspection report"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )

    assert [name for _, _, name in downloads] == list(FILES)
    assert inspections == [(REPO, receipt["source_revision"], "mtp")]


def test_rejects_nonempty_cache_and_download_outside_it(tmp_path: Path):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, inspect, downloads, _ = _fakes(sources, _report(receipt))
    cache = tmp_path / "nonempty-cache"
    cache.mkdir()
    (cache / "prior-file").write_text("not dedicated", encoding="utf-8")

    with pytest.raises(ReceiptExecutionError, match="dedicated cache must be empty"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=cache,
            downloader=download,
            inspector=inspect,
        )
    assert downloads == []

    def outside_download(**kwargs: Any) -> Path:
        target = tmp_path / "outside" / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(sources[kwargs["filename"]])
        return target

    with pytest.raises(
        ReceiptExecutionError,
        match="downloaded source escaped the dedicated cache",
    ):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "empty-cache",
            downloader=outside_download,
            inspector=inspect,
        )


def test_missing_optional_dependency_has_stable_unavailable_reason(tmp_path: Path):
    receipt_path, config_path, _, _ = _fixture(tmp_path)

    def missing_dependency(**kwargs: Any) -> Path:
        del kwargs
        raise ModuleNotFoundError(
            "private import path",
            name="huggingface_hub",
        )

    with pytest.raises(ReceiptUnavailableError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=missing_dependency,
        )

    assert captured.value.reason_code == "missing_optional_dependency"
    assert isinstance(captured.value, ReceiptVerificationError)
    assert captured.value.errors == [
        "source download dependency unavailable for config.json"
    ]


def test_transient_download_has_stable_unavailable_reason(tmp_path: Path):
    receipt_path, config_path, _, _ = _fixture(tmp_path)

    def timed_out(**kwargs: Any) -> Path:
        del kwargs
        raise TimeoutError("request timed out")

    with pytest.raises(ReceiptUnavailableError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=timed_out,
        )

    assert captured.value.reason_code == "source_unavailable"
    assert captured.value.errors == ["source download unavailable for config.json"]


def test_local_permission_failure_is_a_sanitized_execution_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path, config_path, _, _ = _fixture(tmp_path)

    def denied(**kwargs: Any) -> Path:
        del kwargs
        raise PermissionError("/private/cache/path")

    with pytest.raises(ReceiptExecutionError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=denied,
        )

    assert str(captured.value) == (
        "source download execution failed for config.json: PermissionError"
    )
    assert "/private/cache/path" not in str(captured.value)

    assert main(
        ["--receipt", str(receipt_path), "--config", str(config_path), "--json"],
        downloader=denied,
    ) == 4
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "schema_version": 1,
        "status": "error",
        "error_type": "ReceiptExecutionError",
    }
    assert "Traceback" not in output.out
    assert "/private/cache/path" not in output.out


def test_local_input_read_failure_is_an_error_without_path_leakage(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    receipt_path, config_path, _, _ = _fixture(tmp_path)
    original_read_bytes = Path.read_bytes

    def denied(path: Path) -> bytes:
        if path == receipt_path:
            raise PermissionError("/private/receipt/path")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", denied)

    with pytest.raises(ReceiptExecutionError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
        )

    assert str(captured.value) == "receipt read failed: PermissionError"
    assert "/private/receipt/path" not in str(captured.value)
    assert main(
        ["--receipt", str(receipt_path), "--config", str(config_path), "--json"]
    ) == 4
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out) == {
        "schema_version": 1,
        "status": "error",
        "error_type": "ReceiptExecutionError",
    }
    assert "/private/receipt/path" not in output.out


def test_local_hash_failure_is_a_sanitized_execution_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source.json"
    source.write_bytes(b"fixture")

    def denied(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise PermissionError("/private/download/cache")

    monkeypatch.setattr(Path, "open", denied)

    with pytest.raises(ReceiptExecutionError) as captured:
        verifier._hash(source)

    assert str(captured.value) == "source hash execution failed: PermissionError"
    assert "/private/download/cache" not in str(captured.value)


@pytest.mark.parametrize("mode", ["raised", "returned"])
def test_local_missing_download_path_is_an_execution_error(
    tmp_path: Path,
    mode: str,
):
    receipt_path, config_path, _, _ = _fixture(tmp_path)

    def missing(**kwargs: Any) -> Path:
        del kwargs
        missing_path = Path("/private/local-cache/missing")
        if mode == "raised":
            raise FileNotFoundError(str(missing_path))
        return missing_path

    with pytest.raises(ReceiptExecutionError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=missing,
        )

    assert "/private/local-cache" not in str(captured.value)


@pytest.mark.parametrize("error_type", [ImportError, TypeError, ValueError])
def test_injected_inspector_programming_errors_are_execution_errors(
    tmp_path: Path,
    error_type: type[Exception],
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, _, _, _ = _fakes(sources, _report(receipt))

    def broken(**kwargs: Any) -> Any:
        del kwargs
        raise error_type("adapter bug at /private/inspector.py")

    with pytest.raises(ReceiptExecutionError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=broken,
        )

    assert type(captured.value) is ReceiptExecutionError
    assert "/private/inspector.py" not in str(captured.value)


def test_default_inspector_value_error_is_sanitized_semantic_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, _, _, _ = _fakes(sources, _report(receipt))

    def malformed(**kwargs: Any) -> Any:
        del kwargs
        raise ValueError("malformed metadata at /private/hub-cache")

    monkeypatch.setattr(verifier, "_inspect", malformed)

    with pytest.raises(ReceiptVerificationError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
        )

    assert type(captured.value) is ReceiptVerificationError
    assert captured.value.errors == [
        "selected-header inspection found malformed pinned metadata"
    ]
    assert "/private/hub-cache" not in str(captured.value)


@pytest.mark.parametrize("status_code", [408, 425, 429, 500, 503, 599])
def test_transient_http_status_is_unavailable(
    tmp_path: Path,
    status_code: int,
):
    receipt_path, config_path, _, _ = _fixture(tmp_path)

    class Response:
        def __init__(self, status: int):
            self.status_code = status

    class HubHTTPError(RuntimeError):
        def __init__(self, status: int):
            self.response = Response(status)
            super().__init__("signed endpoint at /private/proxy")

    def unavailable(**kwargs: Any) -> Path:
        del kwargs
        raise HubHTTPError(status_code)

    with pytest.raises(ReceiptUnavailableError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=unavailable,
        )

    assert captured.value.reason_code == "source_unavailable"
    assert captured.value.errors == ["source download unavailable for config.json"]
    assert "/private/proxy" not in str(captured.value)


def test_transient_inspection_has_stable_unavailable_reason(tmp_path: Path):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, _, _, _ = _fakes(sources, _report(receipt))

    def timed_out(**kwargs: Any) -> Any:
        del kwargs
        raise TimeoutError("header request timed out")

    with pytest.raises(ReceiptUnavailableError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=timed_out,
        )

    assert captured.value.reason_code == "inspection_unavailable"
    assert captured.value.errors == ["selected-header inspection unavailable"]


def test_default_inspector_transport_error_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, _, _, _ = _fakes(sources, _report(receipt))

    def timed_out(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise TimeoutError("private signed header endpoint timed out")

    monkeypatch.setattr(
        verifier,
        "inspect_deepseek_hub_checkpoint_namespace",
        timed_out,
    )

    with pytest.raises(ReceiptUnavailableError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
        )

    assert captured.value.reason_code == "inspection_unavailable"
    assert captured.value.errors == ["selected-header inspection unavailable"]
    assert "/private" not in str(captured.value)


def test_local_hub_cache_miss_is_unavailable_not_remote_absence(tmp_path: Path):
    receipt_path, config_path, _, _ = _fixture(tmp_path)
    local_cache_miss = type("LocalEntryNotFoundError", (FileNotFoundError,), {})

    def unavailable_locally(**kwargs: Any) -> Path:
        del kwargs
        raise local_cache_miss("not present in the local cache")

    with pytest.raises(ReceiptUnavailableError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=unavailable_locally,
        )

    assert captured.value.reason_code == "source_unavailable"


@pytest.mark.parametrize(
    "source_error",
    [
        type("RepositoryNotFoundError", (RuntimeError,), {}),
        type("RevisionNotFoundError", (RuntimeError,), {}),
        type("RemoteEntryNotFoundError", (RuntimeError,), {}),
    ],
)
def test_pinned_source_absence_is_semantic_failure(
    tmp_path: Path,
    source_error: type[BaseException],
):
    receipt_path, config_path, _, _ = _fixture(tmp_path)

    def absent(**kwargs: Any) -> Path:
        del kwargs
        try:
            raise source_error("pinned source absent")
        except BaseException as exc:
            raise RuntimeError("source resolution failed") from exc

    with pytest.raises(ReceiptVerificationError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=absent,
        )

    assert type(captured.value) is ReceiptVerificationError
    assert captured.value.errors == ["pinned source is absent for config.json"]


def test_pinned_revision_absence_during_inspection_is_semantic_failure(tmp_path: Path):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, _, _, _ = _fakes(sources, _report(receipt))
    revision_not_found = type("RevisionNotFoundError", (RuntimeError,), {})

    def absent(**kwargs: Any) -> Any:
        del kwargs
        try:
            raise revision_not_found("pinned revision absent")
        except RuntimeError as exc:
            raise RuntimeError("model info resolution failed") from exc

    with pytest.raises(ReceiptVerificationError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=absent,
        )

    assert type(captured.value) is ReceiptVerificationError
    assert captured.value.errors == [
        "selected-header inspection found pinned source absence"
    ]


def test_untyped_inspection_rejection_is_a_sanitized_execution_error(tmp_path: Path):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, _, _, _ = _fakes(sources, _report(receipt))

    def rejected(**kwargs: Any) -> Any:
        del kwargs
        raise RuntimeError("immutable source revision was not returned")

    with pytest.raises(ReceiptExecutionError) as captured:
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=rejected,
        )

    assert str(captured.value) == (
        "selected-header inspection execution failed: RuntimeError"
    )
    assert "immutable source revision" not in str(captured.value)


def test_rejects_cache_inside_package_source_or_installation(tmp_path: Path):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, inspect, downloads, _ = _fakes(sources, _report(receipt))

    with pytest.raises(ReceiptExecutionError, match="outside the package"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=(
                Path(verifier.__file__).resolve().parents[1]
                / ".verifier-cache-inside-package-test"
            ),
            downloader=download,
            inspector=inspect,
        )

    assert downloads == []


def test_main_requires_paired_custom_inputs(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as error:
        main(["--receipt", "receipt.json"])

    assert error.value.code == 2
    assert "--receipt and --config must be supplied together" in capsys.readouterr().err


def test_main_supports_text_and_json_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    download, inspect, _, _ = _fakes(sources, _report(receipt))
    assert main(
        ["--receipt", str(receipt_path), "--config", str(config_path)],
        downloader=download,
        inspector=inspect,
    ) == 0
    output = capsys.readouterr().out
    assert "4 source hashes" in output
    assert "0 cached .safetensors files" in output

    receipt_path, config_path, sources, receipt = _fixture(tmp_path / "second")
    report = _report(receipt)
    report.namespace.inventory_sha256 = "0" * 64
    download, inspect, _, _ = _fakes(sources, report)
    assert main(
        ["--receipt", str(receipt_path), "--config", str(config_path), "--json"],
        downloader=download,
        inspector=inspect,
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert any("dspark_namespace" in error for error in payload["errors"])


def test_main_reports_unavailability_as_incomplete_without_raw_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path, config_path, _, _ = _fixture(tmp_path)

    def timed_out(**kwargs: Any) -> Path:
        del kwargs
        raise TimeoutError("request timed out")

    assert main(
        ["--receipt", str(receipt_path), "--config", str(config_path), "--json"],
        downloader=timed_out,
    ) == 3
    assert json.loads(capsys.readouterr().out) == {
        "schema_version": 1,
        "status": "incomplete",
        "reason_code": "source_unavailable",
    }


def test_main_serializes_malformed_report_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    del report.resolved_revision
    download, inspect, _, _ = _fakes(sources, report)

    assert main(
        ["--receipt", str(receipt_path), "--config", str(config_path), "--json"],
        downloader=download,
        inspector=inspect,
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert any("malformed inspection report" in error for error in payload["errors"])


def test_main_serializes_malformed_receipt_without_network_calls(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    receipt["architecture"].pop("dspark_markov_rank")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    download, inspect, downloads, _ = _fakes(sources, report)

    assert main(
        ["--receipt", str(receipt_path), "--config", str(config_path), "--json"],
        downloader=download,
        inspector=inspect,
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert any("architecture field set drift" in error for error in payload["errors"])
    assert downloads == []


@pytest.mark.parametrize("case", ["none", "missing_namespace", "bad_errors"])
def test_malformed_inspection_report_is_always_structured(
    tmp_path: Path,
    case: str,
):
    receipt_path, config_path, sources, receipt = _fixture(tmp_path)
    report = _report(receipt)
    if case == "none":
        report = None
    elif case == "missing_namespace":
        del report.namespace
    else:
        report.namespace.errors = None
    download, inspect, _, _ = _fakes(sources, report)

    with pytest.raises(ReceiptVerificationError, match="malformed inspection report"):
        verify_flash_0731_receipt(
            receipt_path,
            config_path,
            cache_dir=tmp_path / "cache",
            downloader=download,
            inspector=inspect,
        )
