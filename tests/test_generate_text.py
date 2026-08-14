from __future__ import annotations

import hashlib
import io
import json
import shutil
import sys
import types
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast

import pytest
import torch

import nano_deepseek_v4.generate_text as generate_module
from nano_deepseek_v4 import ByteTokenizer, DeepSeekV4ForCausalLM
from nano_deepseek_v4.generate_text import (
    _materialize_remote_bundle,
    _validate_remote_manifest,
    download_huggingface_bundle,
    main,
    run_text_generation,
)
from nano_deepseek_v4.train_text import tiny_text_config


class _GenerationArguments(TypedDict):
    prompt: str
    max_new_tokens: int
    device: str
    temperature: float
    top_p: float
    seed: int


class _FakeHubModule(types.ModuleType):
    HfApi: type[object]
    hf_hub_download: Callable[..., str]

    def __init__(
        self,
        *,
        api: type[object],
        download: Callable[..., str],
    ) -> None:
        super().__init__("huggingface_hub")
        self.HfApi = api
        self.hf_hub_download = download


def _save_bundle(path: Path, *, with_tokenizer: bool = True, max_positions: int = 64) -> Path:
    torch.manual_seed(17)
    config = tiny_text_config()
    config.max_position_embeddings = max_positions
    model = DeepSeekV4ForCausalLM(config).eval()
    model.save_pretrained(
        path,
        tokenizer=ByteTokenizer() if with_tokenizer else None,
    )
    return path


def test_local_generation_is_deterministic_and_receipted(tmp_path: Path):
    bundle = _save_bundle(tmp_path / "bundle")

    first = run_text_generation(
        bundle,
        prompt="ROMEO:",
        max_new_tokens=3,
        device="cpu",
    )
    second = run_text_generation(
        bundle,
        prompt="ROMEO:",
        max_new_tokens=3,
        device="cpu",
    )
    tokenizer = ByteTokenizer()
    direct_model = DeepSeekV4ForCausalLM.from_pretrained(bundle).eval()
    direct_prompt = tokenizer.encode("ROMEO:").unsqueeze(0)
    direct = direct_model.generate(
        direct_prompt,
        max_new_tokens=3,
        stop_on_eos=False,
    )

    assert first.continuation_token_ids == second.continuation_token_ids
    assert first.continuation_token_ids == direct[0, direct_prompt.shape[1] :].tolist()
    assert first.generated_text == second.generated_text
    assert first.schema_version == 1
    assert first.source.kind == "local"
    assert first.prompt_token_count == 6
    assert first.generated_new_tokens == 3
    assert first.parameter_count == 246_590
    assert len(first.bundle_manifest_sha256) == 64
    assert len(first.config_sha256) == 64
    assert len(first.tokenizer_sha256) == 64


def test_seeded_sampling_is_reproducible(tmp_path: Path):
    bundle = _save_bundle(tmp_path / "sampled")
    arguments: _GenerationArguments = {
        "prompt": "Juliet",
        "max_new_tokens": 4,
        "device": "cpu",
        "temperature": 0.8,
        "top_p": 0.9,
        "seed": 123,
    }

    first = run_text_generation(bundle, **arguments)
    second = run_text_generation(bundle, **arguments)

    assert first.continuation_token_ids == second.continuation_token_ids
    assert first.temperature == 0.8
    assert first.top_p == 0.9


def test_generation_rejects_model_only_v1_bundle(tmp_path: Path):
    bundle = _save_bundle(tmp_path / "v1", with_tokenizer=False)

    with pytest.raises(ValueError, match="format_version 2"):
        run_text_generation(bundle, prompt="hello", max_new_tokens=1, device="cpu")


def test_prompt_overflow_is_rejected_before_model_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = _save_bundle(tmp_path / "short", max_positions=8)

    import nano_deepseek_v4.generate_text as generate_module

    monkeypatch.setattr(
        generate_module.DeepSeekV4ForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("model allocation must not happen"),
    )
    with pytest.raises(ValueError, match="exceed config max_position_embeddings"):
        run_text_generation(
            bundle,
            prompt="12345678",
            max_new_tokens=1,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prompt": ""}, "prompt must be non-empty"),
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**64}, "seed"),
        ({"temperature": 0.0}, "temperature"),
        ({"top_p": 0.5}, "top_p only applies"),
    ],
)
def test_generation_argument_validation_precedes_bundle_access(overrides, message):
    arguments = {
        "prompt": "hello",
        "max_new_tokens": 1,
        "device": "cpu",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_text_generation(
            "missing",
            prompt=cast(str, arguments["prompt"]),
            max_new_tokens=cast(int, arguments["max_new_tokens"]),
            device=cast(str, arguments["device"]),
            seed=cast(int, arguments.get("seed", 0)),
            temperature=cast(float | None, arguments.get("temperature")),
            top_p=cast(float, arguments.get("top_p", 1.0)),
        )


def test_cli_emits_json_with_token_ids(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    bundle = _save_bundle(tmp_path / "cli")

    assert (
        main(
            [
                "--bundle",
                str(bundle),
                "--prompt",
                "ROMEO:",
                "--max-new-tokens",
                "2",
                "--device",
                "cpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["source"]["kind"] == "local"
    assert payload["prompt_text"] == "ROMEO:"
    assert len(payload["continuation_token_ids"]) == 2


def test_cli_reads_prompt_from_stdin(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    bundle = _save_bundle(tmp_path / "stdin")
    monkeypatch.setattr(sys, "stdin", io.StringIO("Juliet"))

    assert (
        main(
            [
                "--bundle",
                str(bundle),
                "--max-new-tokens",
                "0",
                "--device",
                "cpu",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["prompt_text"] == "Juliet"
    assert payload["generated_text"] == "Juliet"


def test_cli_requires_revision_for_hub_source():
    with pytest.raises(SystemExit) as exc_info:
        main(["--hf-repo", "owner/model", "--prompt", "hello"])

    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--prompt", "hello", "--max-new-tokens", "-1"],
        ["--prompt", "hello", "--seed", str(2**64)],
    ],
)
def test_hub_cli_validates_generation_arguments_before_download(
    invalid_args: list[str],
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        generate_module,
        "download_huggingface_bundle",
        lambda *args, **kwargs: pytest.fail("Hub download must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--hf-repo", "owner/model", "--revision", "main", *invalid_args])

    assert exc_info.value.code == 2


def test_hub_cli_reads_prompt_file_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        generate_module,
        "download_huggingface_bundle",
        lambda *args, **kwargs: pytest.fail("Hub download must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hf-repo",
                "owner/model",
                "--revision",
                "main",
                "--prompt-file",
                str(tmp_path / "missing.txt"),
            ]
        )

    assert exc_info.value.code == 2


def test_hub_cli_rejects_oversized_prompt_before_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("too large", encoding="utf-8")
    monkeypatch.setattr(generate_module, "_MAX_PROMPT_FILE_BYTES", 1)
    monkeypatch.setattr(
        generate_module,
        "download_huggingface_bundle",
        lambda *args, **kwargs: pytest.fail("Hub download must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--hf-repo",
                "owner/model",
                "--revision",
                "main",
                "--prompt-file",
                str(prompt),
            ]
        )

    assert exc_info.value.code == 2


def test_hub_download_pins_and_reuses_one_resolved_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = "a" * 40
    manifest_path = _save_bundle(tmp_path / "snapshot") / "nano_deepseek_v4.json"
    manifest = json.loads(manifest_path.read_text())
    blob_directory = tmp_path / "blobs"
    blob_directory.mkdir()
    for path in list(manifest_path.parent.iterdir()):
        blob = blob_directory / path.name
        path.replace(blob)
        path.symlink_to(blob)
    calls: list[tuple[str, str]] = []

    class FakeApi:
        def model_info(self, *, repo_id, revision):
            assert repo_id == "owner/model"
            assert revision == "v1"
            return SimpleNamespace(sha=resolved)

    def fake_download(*, repo_id, filename, revision, repo_type, cache_dir):
        assert repo_id == "owner/model"
        assert repo_type == "model"
        assert cache_dir == str(tmp_path / "cache")
        calls.append((filename, revision))
        if filename == "nano_deepseek_v4.json":
            return str(manifest_path)
        return str(manifest_path.parent / filename)

    fake_module = _FakeHubModule(api=FakeApi, download=fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    source = download_huggingface_bundle(
        "owner/model",
        "v1",
        cache_dir=tmp_path / "cache",
    )

    assert source.kind == "huggingface"
    assert source.repo_id == "owner/model"
    assert source.requested_revision == "v1"
    assert source.resolved_revision == resolved
    materialized = Path(source.bundle_path)
    assert materialized != manifest_path.parent
    assert all(not path.is_symlink() for path in materialized.iterdir())
    assert json.loads((materialized / "nano_deepseek_v4.json").read_text()) == manifest
    assert calls[0] == ("nano_deepseek_v4.json", resolved)
    assert all(revision == resolved for _, revision in calls)
    shard_names = sorted(
        set(manifest["sha256"])
        - {
            "config.json",
            "model.safetensors.index.json",
            "nano_deepseek_v4_tokenizer.json",
        }
    )
    assert [filename for filename, _ in calls] == [
        "nano_deepseek_v4.json",
        "config.json",
        "model.safetensors.index.json",
        "nano_deepseek_v4_tokenizer.json",
        *shard_names,
    ]


def test_materialization_reuses_valid_bundle_published_by_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    downloaded = _save_bundle(tmp_path / "downloaded")
    competitor = _save_bundle(tmp_path / "competitor")
    manifest_sha256 = hashlib.sha256(
        (downloaded / "nano_deepseek_v4.json").read_bytes()
    ).hexdigest()

    def publish_competitor_then_fail(source: Path, target: Path) -> None:
        shutil.copytree(competitor, target)
        raise OSError("simulated first-publish collision")

    monkeypatch.setattr(generate_module.os, "replace", publish_competitor_then_fail)

    materialized = _materialize_remote_bundle(
        {path.name: path for path in downloaded.iterdir()},
        repo_id="owner/model",
        resolved_revision="a" * 40,
        manifest_sha256=manifest_sha256,
        cache_dir=tmp_path / "cache",
    )

    assert materialized.is_dir()
    assert hashlib.sha256(
        (materialized / "nano_deepseek_v4.json").read_bytes()
    ).hexdigest() == manifest_sha256
    assert not list(materialized.parent.glob(f".{materialized.name}-*"))


def test_materialization_rejects_invalid_competing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    downloaded = _save_bundle(tmp_path / "downloaded")
    manifest_sha256 = hashlib.sha256(
        (downloaded / "nano_deepseek_v4.json").read_bytes()
    ).hexdigest()

    def publish_invalid_target_then_fail(source: Path, target: Path) -> None:
        target.mkdir()
        raise OSError("simulated first-publish collision")

    monkeypatch.setattr(generate_module.os, "replace", publish_invalid_target_then_fail)

    with pytest.raises(ValueError, match="materialized Hub bundle verification failed"):
        _materialize_remote_bundle(
            {path.name: path for path in downloaded.iterdir()},
            repo_id="owner/model",
            resolved_revision="b" * 40,
            manifest_sha256=manifest_sha256,
            cache_dir=tmp_path / "cache",
        )


def test_hub_download_rejects_metadata_checksum_before_other_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = "b" * 40
    manifest_path = _save_bundle(tmp_path / "snapshot") / "nano_deepseek_v4.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["config.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    calls: list[str] = []

    class FakeApi:
        def model_info(self, *, repo_id, revision):
            return SimpleNamespace(sha=resolved)

    def fake_download(*, repo_id, filename, revision, repo_type, cache_dir):
        calls.append(filename)
        return str(manifest_path.parent / filename)

    fake_module = _FakeHubModule(api=FakeApi, download=fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    with pytest.raises(ValueError, match="checksum mismatch.*config.json"):
        download_huggingface_bundle("owner/model", "v1")

    assert calls == ["nano_deepseek_v4.json", "config.json"]


def test_hub_download_validates_metadata_contract_before_weight_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    resolved = "c" * 40
    manifest_path = _save_bundle(tmp_path / "snapshot") / "nano_deepseek_v4.json"
    config_path = manifest_path.parent / "config.json"
    config_path.write_text("{}\n")
    manifest = json.loads(manifest_path.read_text())
    manifest["sha256"]["config.json"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    calls: list[str] = []

    class FakeApi:
        def model_info(self, *, repo_id, revision):
            return SimpleNamespace(sha=resolved)

    def fake_download(*, repo_id, filename, revision, repo_type, cache_dir):
        calls.append(filename)
        return str(manifest_path.parent / filename)

    fake_module = _FakeHubModule(api=FakeApi, download=fake_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    with pytest.raises(ValueError, match="vocab_size"):
        download_huggingface_bundle("owner/model", "v1")

    assert calls == [
        "nano_deepseek_v4.json",
        "config.json",
        "model.safetensors.index.json",
        "nano_deepseek_v4_tokenizer.json",
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest["sha256"].update({"../escape": "4" * 64}),
            "unsafe remote bundle filename",
        ),
        (
            lambda manifest: manifest["sha256"].update(
                {"model-00001-of-00001.safetensors": "A" * 64}
            ),
            "invalid SHA-256",
        ),
        (
            lambda manifest: manifest["sha256"].update(
                {"model-00002-of-00003.safetensors": "4" * 64}
            ),
            "disagree|incomplete",
        ),
    ],
)
def test_remote_manifest_rejects_unsafe_inventory(tmp_path: Path, mutate, message):
    manifest = {
        "format": "nano-deepseek-v4-pretrained",
        "format_version": 2,
        "model_class": "DeepSeekV4ForCausalLM",
        "config_file": "config.json",
        "weights_index": "model.safetensors.index.json",
        "tokenizer_file": "nano_deepseek_v4_tokenizer.json",
        "sha256": {
            "config.json": "0" * 64,
            "model.safetensors.index.json": "1" * 64,
            "nano_deepseek_v4_tokenizer.json": "2" * 64,
            "model-00001-of-00001.safetensors": "3" * 64,
        },
    }
    mutate(manifest)
    path = tmp_path / "nano_deepseek_v4.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        _validate_remote_manifest(path)
