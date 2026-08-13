from __future__ import annotations

import hashlib
import json
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from nano_deepseek_v4.architecture import main
from nano_deepseek_v4.checkpoint import (
    _SAFETENSORS_DTYPE_BYTES,
    _expected_official_tensor_shapes,
    _shape_numel,
    inspect_deepseek_hub_checkpoint_namespace,
)
from nano_deepseek_v4.config import DeepSeekV4Config

_RESOLVED_REVISION = "0123456789abcdef0123456789abcdef01234567"
_REPO_ID = "example/deepseek-v4"


def _official_config() -> dict[str, Any]:
    return {
        "vocab_size": 16,
        "hidden_size": 8,
        "moe_intermediate_size": 12,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "q_lora_rank": 4,
        "num_experts_per_tok": 1,
        "n_routed_experts": 2,
        "n_shared_experts": 1,
        "routed_scaling_factor": 1.5,
        "max_position_embeddings": 32,
        "compress_rate_csa": 2,
        "compress_rate_hca": 4,
        "compress_ratios": [0, 0],
        "num_hash_layers": 0,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 8,
        "sliding_window": 4,
        "o_groups": 1,
        "o_lora_rank": 4,
        "index_n_heads": 2,
        "index_head_dim": 4,
        "index_topk": 2,
        "num_nextn_predict_layers": 1,
        "qk_rope_head_dim": 2,
    }


@dataclass(frozen=True)
class _FakeTensorInfo:
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


class _FakeHubModule(types.ModuleType):
    __version__: str
    HfApi: type[object]
    hf_hub_download: Callable[..., str]

    def __init__(
        self,
        *,
        version: str,
        api: type[object],
        download: Callable[..., str],
    ) -> None:
        super().__init__("huggingface_hub")
        self.__version__ = version
        self.HfApi = api
        self.hf_hub_download = download


def _tensor_metadata(
    keys: list[str],
    shapes: dict[str, tuple[int, ...]],
) -> dict[str, _FakeTensorInfo]:
    offset = 0
    result: dict[str, _FakeTensorInfo] = {}
    for key in sorted(keys):
        size = _shape_numel(shapes[key]) * _SAFETENSORS_DTYPE_BYTES["F32"]
        result[key] = _FakeTensorInfo("F32", shapes[key], (offset, offset + size))
        offset += size
    return result


def _install_fake_hub(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    tensors: dict[str, _FakeTensorInfo] | None = None,
    index_mutator: Any = None,
    parse_error: BaseException | None = None,
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    official = _official_config()
    config = DeepSeekV4Config.from_official_json(official)
    shapes = _expected_official_tensor_shapes(config)
    mtp_keys = sorted(key for key in shapes if key.startswith("mtp."))
    backbone_keys = sorted(set(shapes) - set(mtp_keys))
    weight_map = {
        **{key: "model-00001-of-00002.safetensors" for key in backbone_keys},
        **{key: "model-00002-of-00002.safetensors" for key in mtp_keys},
    }
    index: dict[str, Any] = {
        "metadata": {"total_size": sum(_shape_numel(shape) * 4 for shape in shapes.values())},
        "weight_map": weight_map,
    }
    if index_mutator is not None:
        index_mutator(index)

    config_path = tmp_path / "config.json"
    index_path = tmp_path / "model.safetensors.index.json"
    config_path.write_text(json.dumps(official, sort_keys=True))
    index_path.write_text(json.dumps(index, sort_keys=True))
    mtp_tensors = tensors if tensors is not None else _tensor_metadata(mtp_keys, shapes)
    parse_calls: list[tuple[str, str, str]] = []
    model_info_calls: list[tuple[str, str]] = []
    download_calls: list[tuple[str, str, str]] = []

    class FakeHfApi:
        def __init__(self, token: str | bool | None = None):
            self.token = token

        def model_info(self, *, repo_id: str, revision: str):
            model_info_calls.append((repo_id, revision))
            return types.SimpleNamespace(sha=_RESOLVED_REVISION.upper())

        def parse_safetensors_file_metadata(
            self,
            *,
            repo_id: str,
            filename: str,
            repo_type: str,
            revision: str,
            token: str | bool | None,
        ):
            assert repo_type == "model"
            parse_calls.append((repo_id, filename, revision))
            if parse_error is not None:
                raise parse_error
            if filename != "model-00002-of-00002.safetensors":
                raise AssertionError(f"unneeded shard header requested: {filename}")
            return types.SimpleNamespace(tensors=mtp_tensors)

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
        revision: str,
        token: str | bool | None,
        cache_dir: str | None = None,
    ) -> str:
        assert repo_type == "model"
        assert token is None
        download_calls.append((repo_id, filename, revision))
        return str(config_path if filename == "config.json" else index_path)

    fake_module = _FakeHubModule(
        version="0.34.0",
        api=FakeHfApi,
        download=fake_hf_hub_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)
    calls = {
        "model_info": model_info_calls,
        "downloads": download_calls,
        "parse": parse_calls,
        "index_path": index_path,
        "config_path": config_path,
        "mtp_tensors": mtp_tensors,
    }
    return calls, parse_calls


def test_hub_namespace_inspection_pins_revision_and_reads_only_selected_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls, parse_calls = _install_fake_hub(monkeypatch, tmp_path)

    report = inspect_deepseek_hub_checkpoint_namespace(
        _REPO_ID,
        "main",
        "mtp",
        cache_dir=tmp_path / "cache",
    )

    assert report.is_complete
    assert report.metadata_only
    assert report.requested_revision == "main"
    assert report.resolved_revision == _RESOLVED_REVISION
    assert report.huggingface_hub_version == "0.34.0"
    assert calls["model_info"] == [(_REPO_ID, "main")]
    assert calls["downloads"] == [
        (_REPO_ID, "config.json", _RESOLVED_REVISION),
        (_REPO_ID, "model.safetensors.index.json", _RESOLVED_REVISION),
    ]
    assert parse_calls == [
        (_REPO_ID, "model-00002-of-00002.safetensors", _RESOLVED_REVISION)
    ]
    assert report.inspected_shard_count == 1
    assert report.total_shard_count == 2
    assert report.namespace.snapshot_preflight_complete is False
    assert report.namespace.shard_files == ["model-00002-of-00002.safetensors"]
    expected_inventory = {
        key: {"dtype": tensor.dtype, "shape": list(tensor.shape)}
        for key, tensor in calls["mtp_tensors"].items()
    }
    canonical = json.dumps(expected_inventory, sort_keys=True, separators=(",", ":"))
    assert report.namespace.inventory_sha256 == hashlib.sha256(canonical.encode()).hexdigest()
    assert report.index_sha256 == hashlib.sha256(
        calls["index_path"].read_bytes()
    ).hexdigest()
    assert report.config_sha256 == hashlib.sha256(
        calls["config_path"].read_bytes()
    ).hexdigest()
    assert "config" not in report.to_dict()


def test_hub_namespace_inspection_propagates_header_transport_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    error = TimeoutError("private signed header endpoint timed out")
    _install_fake_hub(monkeypatch, tmp_path, parse_error=error)

    with pytest.raises(TimeoutError) as captured:
        inspect_deepseek_hub_checkpoint_namespace(
            _REPO_ID,
            "main",
            "mtp",
            cache_dir=tmp_path / "cache",
        )

    assert captured.value is error


@pytest.mark.parametrize(
    ("mutation", "expected_field", "expected_text"),
    [
        ("missing", "missing_keys_in_shards", "mtp."),
        ("unexpected", "unexpected_keys_in_shards", "mtp.99.extra"),
        ("shape", "shape_mismatches", "expected shape"),
        ("offsets", "errors", "data offsets span"),
        ("overlap", "errors", "overlaps"),
        ("dtype", "errors", "missing scale sidecar"),
    ],
)
def test_hub_namespace_inspection_rejects_corrupt_header_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_field: str,
    expected_text: str,
):
    config = DeepSeekV4Config.from_official_json(_official_config())
    shapes = _expected_official_tensor_shapes(config)
    mtp_keys = sorted(key for key in shapes if key.startswith("mtp."))
    tensors = _tensor_metadata(mtp_keys, shapes)
    target = next(key for key in mtp_keys if not key.endswith(".scale"))
    if mutation == "missing":
        tensors.pop(target)
    elif mutation == "unexpected":
        tensors["mtp.99.extra"] = _FakeTensorInfo("F32", (1,), (0, 4))
    elif mutation == "shape":
        tensor = tensors[target]
        tensors[target] = replace(tensor, shape=(1,))
    elif mutation == "offsets":
        tensor = tensors[target]
        tensors[target] = replace(tensor, data_offsets=(0, 1))
    elif mutation == "overlap":
        target = mtp_keys[1]
        tensor = tensors[target]
        tensors[target] = replace(
            tensor,
            data_offsets=(0, tensor.data_offsets[1] - tensor.data_offsets[0]),
        )
    else:
        target = "mtp.0.enorm.weight"
        tensor = tensors[target]
        tensors[target] = replace(
            tensor,
            dtype="I8",
            data_offsets=(0, _shape_numel(tensor.shape)),
        )
    _install_fake_hub(monkeypatch, tmp_path, tensors=tensors)

    report = inspect_deepseek_hub_checkpoint_namespace(_REPO_ID, "main", "mtp")

    assert not report.is_complete
    values = getattr(report.namespace, expected_field)
    assert any(expected_text in value for value in values)


def test_hub_namespace_inspection_rejects_unsafe_index_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def mutate(index: dict[str, Any]) -> None:
        key = next(key for key in index["weight_map"] if key.startswith("mtp."))
        index["weight_map"][key] = "../weights.safetensors"

    _install_fake_hub(monkeypatch, tmp_path, index_mutator=mutate)

    with pytest.raises(ValueError, match="Unsafe safetensors shard path"):
        inspect_deepseek_hub_checkpoint_namespace(_REPO_ID, "main", "mtp")


@pytest.mark.parametrize(
    ("repo_id", "revision", "namespace", "message"),
    [
        ("missing-owner", "main", "mtp", "owner/name"),
        (_REPO_ID, " ", "mtp", "revision"),
        (_REPO_ID, "main", "../mtp", "dot-separated identifier"),
    ],
)
def test_hub_namespace_inspection_validates_coordinates_before_import(
    repo_id: str,
    revision: str,
    namespace: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        inspect_deepseek_hub_checkpoint_namespace(repo_id, revision, namespace)


def test_hub_namespace_inspection_has_actionable_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ImportError, match=r"nano-deepseek-v4\[official\]"):
        inspect_deepseek_hub_checkpoint_namespace(_REPO_ID, "main", "mtp")


def test_hub_inspection_cli_emits_scope_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _install_fake_hub(monkeypatch, tmp_path)

    return_code = main(
        [
            "--hf-repo",
            _REPO_ID,
            "--revision",
            "main",
            "--namespace",
            "mtp",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["source"] == f"hf:{_REPO_ID}@{_RESOLVED_REVISION}"
    assert payload["hub_checkpoint"]["metadata_only"]
    assert payload["hub_checkpoint"]["resolved_revision"] == _RESOLVED_REVISION
    assert payload["namespace"]["snapshot_preflight_complete"] is False
    assert payload["namespace"]["is_complete"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--hf-repo", _REPO_ID, "--revision", "main"],
        ["--hf-repo", _REPO_ID, "--namespace", "mtp"],
        ["--revision", "main"],
        ["--hf-cache-dir", "cache"],
    ],
)
def test_hub_inspection_cli_requires_complete_hub_coordinates(arguments: list[str]):
    with pytest.raises(SystemExit):
        main(arguments)
