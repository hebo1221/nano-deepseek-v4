from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import (
    load_deepseek_v4_pretrained_tokenizer,
    verify_deepseek_v4_pretrained_bundle,
)
from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM
from .tokenizer import BYTE_TOKENIZER_FILENAME, ByteTokenizer

_MANIFEST_NAME = "nano_deepseek_v4.json"
_CONFIG_NAME = "config.json"
_INDEX_NAME = "model.safetensors.index.json"
_TOKENIZER_NAME = BYTE_TOKENIZER_FILENAME
_MAX_REMOTE_MANIFEST_BYTES = 1024 * 1024
_MAX_REMOTE_CONFIG_BYTES = 1024 * 1024
_MAX_REMOTE_INDEX_BYTES = 64 * 1024 * 1024
_MAX_REMOTE_BUNDLE_FILES = 4096
_MAX_PROMPT_FILE_BYTES = 16 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHARD_PATTERN = re.compile(r"model-(\d{5})-of-(\d{5})\.safetensors")


@dataclass(frozen=True)
class BundleSource:
    kind: str
    bundle_path: str
    repo_id: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    schema_version: int
    source: BundleSource
    bundle_manifest_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    device: str
    dtype: str
    parameter_count: int
    seed: int
    temperature: float | None
    top_p: float
    stop_on_eos: bool
    prompt_text: str
    prompt_token_count: int
    max_new_tokens: int
    generated_new_tokens: int
    continuation_token_ids: list[int]
    generated_text: str
    continuation_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    if resolved.type == "meta":
        raise ValueError("generation requires a materialized device, not meta.")
    return resolved


def _resolve_dtype(dtype: str | torch.dtype | None) -> tuple[torch.dtype | None, str]:
    if dtype is None or dtype == "auto":
        return None, "stored"
    if isinstance(dtype, torch.dtype):
        supported = {
            torch.float32: "float32",
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
        }
        if dtype not in supported:
            raise ValueError("dtype must be float32, bfloat16, float16, or auto.")
        return dtype, supported[dtype]
    aliases = {
        "float32": (torch.float32, "float32"),
        "bfloat16": (torch.bfloat16, "bfloat16"),
        "float16": (torch.float16, "float16"),
    }
    if dtype not in aliases:
        raise ValueError("dtype must be float32, bfloat16, float16, or auto.")
    return aliases[dtype]


def _validate_generation_arguments(
    *,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    temperature: float | None,
    top_p: float,
    stop_on_eos: bool,
) -> None:
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be non-empty text.")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError("max_new_tokens must be a non-negative integer.")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > 2**64 - 1
    ):
        raise ValueError("seed must be an integer in [0, 2**64 - 1].")
    if temperature is not None:
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise ValueError("temperature must be a finite positive number or None.")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0 < float(top_p) <= 1
    ):
        raise ValueError("top_p must be in (0, 1].")
    if temperature is None and float(top_p) != 1.0:
        raise ValueError("top_p only applies when temperature enables sampling.")
    if not isinstance(stop_on_eos, bool):
        raise TypeError("stop_on_eos must be a bool.")


def _validate_remote_manifest(path: Path) -> dict[str, str]:
    if path.stat().st_size > _MAX_REMOTE_MANIFEST_BYTES:
        raise ValueError("remote native manifest exceeds the 1 MiB safety limit.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("remote native manifest is not valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("remote native manifest root must be an object.")
    expected_manifest_keys = {
        "format",
        "format_version",
        "model_class",
        "config_file",
        "weights_index",
        "tokenizer_file",
        "sha256",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("remote native manifest has an invalid key inventory.")
    expected_values = {
        "format": "nano-deepseek-v4-pretrained",
        "format_version": 2,
        "model_class": "DeepSeekV4ForCausalLM",
        "config_file": _CONFIG_NAME,
        "weights_index": _INDEX_NAME,
        "tokenizer_file": _TOKENIZER_NAME,
    }
    for name, expected in expected_values.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"remote native manifest has unsupported {name}: {manifest.get(name)!r}"
            )
    checksums = manifest.get("sha256")
    if not isinstance(checksums, dict):
        raise ValueError("remote native manifest sha256 must be an object.")
    if not 4 <= len(checksums) <= _MAX_REMOTE_BUNDLE_FILES:
        raise ValueError("remote native manifest has an unsafe file count.")
    names = set(checksums)
    required = {_CONFIG_NAME, _INDEX_NAME, _TOKENIZER_NAME}
    if not required.issubset(names):
        raise ValueError("remote native manifest is missing required metadata files.")
    shard_names = names - required
    if not shard_names:
        raise ValueError("remote native manifest does not declare any weight shards.")
    shard_indices: set[int] = set()
    shard_totals: set[int] = set()
    for name, digest in checksums.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "\\" in name
            or any(ord(character) < 32 for character in name)
        ):
            raise ValueError(f"unsafe remote bundle filename: {name!r}")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"invalid SHA-256 digest for remote bundle file {name!r}.")
        if name in required:
            continue
        match = _SHARD_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"unsupported remote weight shard filename: {name!r}")
        shard_indices.add(int(match.group(1)))
        shard_totals.add(int(match.group(2)))
    if len(shard_totals) != 1:
        raise ValueError("remote weight shard names disagree on total shard count.")
    total_shards = next(iter(shard_totals))
    if total_shards != len(shard_names) or shard_indices != set(range(1, total_shards + 1)):
        raise ValueError("remote weight shard sequence is incomplete or inconsistent.")
    return {name: checksums[name] for name in sorted(names)}


def _verify_remote_download(
    path: Path,
    *,
    filename: str,
    expected_sha256: str,
    snapshot_directory: Path,
) -> None:
    if path.name != filename or path.parent.resolve() != snapshot_directory.resolve():
        raise ValueError(
            f"Hugging Face returned an unexpected path for {filename!r}: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"downloaded remote bundle file is missing: {filename}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"remote bundle checksum mismatch for {filename!r}: "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def _validate_remote_metadata(
    paths: dict[str, Path],
    checksums: dict[str, str],
) -> list[str]:
    config_path = paths[_CONFIG_NAME]
    index_path = paths[_INDEX_NAME]
    tokenizer_path = paths[_TOKENIZER_NAME]
    if config_path.stat().st_size > _MAX_REMOTE_CONFIG_BYTES:
        raise ValueError("remote native config exceeds the 1 MiB safety limit.")
    if index_path.stat().st_size > _MAX_REMOTE_INDEX_BYTES:
        raise ValueError("remote native weight index exceeds the 64 MiB safety limit.")

    config = DeepSeekV4Config.from_json_file(config_path)
    tokenizer = ByteTokenizer.from_json_file(tokenizer_path)
    tokenizer.validate_config(config)

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("remote native weight index is not valid JSON.") from exc
    if not isinstance(index, dict) or set(index) != {"metadata", "weight_map"}:
        raise ValueError("remote native weight index has an invalid key inventory.")
    metadata = index["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != {"format", "total_size"}:
        raise ValueError("remote native weight index metadata is invalid.")
    total_size = metadata["total_size"]
    if (
        metadata["format"] != "pt"
        or isinstance(total_size, bool)
        or not isinstance(total_size, int)
        or total_size <= 0
    ):
        raise ValueError("remote native weight index metadata is invalid.")

    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("remote native weight index must contain a non-empty weight_map.")
    declared_shards = set(checksums) - {_CONFIG_NAME, _INDEX_NAME, _TOKENIZER_NAME}
    mapped_shards: set[str] = set()
    for key, shard in weight_map.items():
        if not isinstance(key, str) or not key or any(ord(character) < 32 for character in key):
            raise ValueError("remote native weight index contains an invalid tensor key.")
        if not isinstance(shard, str) or shard not in declared_shards:
            raise ValueError(
                f"remote native weight index maps {key!r} to an undeclared shard."
            )
        mapped_shards.add(shard)
    if mapped_shards != declared_shards:
        raise ValueError(
            "remote native manifest and weight index disagree on the shard inventory."
        )
    return sorted(declared_shards)


def _materialized_bundle_directory(
    *,
    repo_id: str,
    resolved_revision: str,
    manifest_sha256: str,
    cache_dir: str | Path | None,
) -> Path:
    if cache_dir is not None:
        root = Path(cache_dir) / "nano-deepseek-v4-bundles"
    else:
        default_hf_home = Path.home() / ".cache" / "huggingface"
        hf_home = Path(os.environ.get("HF_HOME", default_hf_home))
        root = hf_home / "nano-deepseek-v4-bundles"
    repository_key = hashlib.sha256(repo_id.encode("utf-8")).hexdigest()
    return root / repository_key / resolved_revision / manifest_sha256


def _verify_materialized_bundle(path: Path, manifest_sha256: str) -> None:
    report = verify_deepseek_v4_pretrained_bundle(path)
    if not report.is_complete or not report.generation_ready:
        details = "; ".join(report.errors) or "bundle is not generation-ready"
        raise ValueError(f"materialized Hub bundle verification failed: {details}")
    if report.manifest_sha256 != manifest_sha256:
        raise ValueError(
            "materialized Hub bundle manifest does not match the resolved repository."
        )


def _materialize_remote_bundle(
    downloaded_paths: dict[str, Path],
    *,
    repo_id: str,
    resolved_revision: str,
    manifest_sha256: str,
    cache_dir: str | Path | None,
) -> Path:
    target = _materialized_bundle_directory(
        repo_id=repo_id,
        resolved_revision=resolved_revision,
        manifest_sha256=manifest_sha256,
        cache_dir=cache_dir,
    )
    if target.exists():
        if not target.is_dir():
            raise ValueError(f"materialized Hub bundle path is not a directory: {target}")
        _verify_materialized_bundle(target, manifest_sha256)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        for filename, source in downloaded_paths.items():
            shutil.copyfile(source, staging / filename)
        _verify_materialized_bundle(staging, manifest_sha256)
        staging.chmod(0o755)
        if target.exists():
            _verify_materialized_bundle(target, manifest_sha256)
            return target
        try:
            os.replace(staging, target)
        except OSError as exc:
            if not target.exists():
                raise
            if not target.is_dir():
                raise ValueError(
                    f"materialized Hub bundle path is not a directory: {target}"
                ) from exc
            _verify_materialized_bundle(target, manifest_sha256)
            return target
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def download_huggingface_bundle(
    repo_id: str,
    revision: str,
    *,
    cache_dir: str | Path | None = None,
) -> BundleSource:
    """Resolve a Hub revision once, then download only a validated v2 inventory."""

    if not repo_id or not revision:
        raise ValueError("repo_id and revision must be non-empty.")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face download support requires "
            "`python -m pip install 'nano-deepseek-v4[official]'`."
        ) from exc

    info = HfApi().model_info(repo_id=repo_id, revision=revision)
    resolved_revision = getattr(info, "sha", None)
    if not isinstance(resolved_revision, str) or _REVISION_PATTERN.fullmatch(
        resolved_revision
    ) is None:
        raise ValueError("Hugging Face did not return an immutable commit revision.")
    cache = None if cache_dir is None else str(cache_dir)
    manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=_MANIFEST_NAME,
            revision=resolved_revision,
            repo_type="model",
            cache_dir=cache,
        )
    )
    checksums = _validate_remote_manifest(manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)
    snapshot_directory = manifest_path.parent
    downloaded_paths = {_MANIFEST_NAME: manifest_path}
    metadata_paths: dict[str, Path] = {}
    for filename in (_CONFIG_NAME, _INDEX_NAME, _TOKENIZER_NAME):
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=resolved_revision,
                repo_type="model",
                cache_dir=cache,
            )
        )
        _verify_remote_download(
            downloaded,
            filename=filename,
            expected_sha256=checksums[filename],
            snapshot_directory=snapshot_directory,
        )
        metadata_paths[filename] = downloaded
        downloaded_paths[filename] = downloaded

    shard_filenames = _validate_remote_metadata(metadata_paths, checksums)
    for filename in shard_filenames:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=resolved_revision,
                repo_type="model",
                cache_dir=cache,
            )
        )
        _verify_remote_download(
            downloaded,
            filename=filename,
            expected_sha256=checksums[filename],
            snapshot_directory=snapshot_directory,
        )
        downloaded_paths[filename] = downloaded
    materialized = _materialize_remote_bundle(
        downloaded_paths,
        repo_id=repo_id,
        resolved_revision=resolved_revision,
        manifest_sha256=manifest_sha256,
        cache_dir=cache_dir,
    )
    return BundleSource(
        kind="huggingface",
        bundle_path=str(materialized),
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
    )


def run_text_generation(
    bundle_directory: str | Path,
    *,
    prompt: str,
    max_new_tokens: int = 80,
    device: str | torch.device = "auto",
    dtype: str | torch.dtype | None = "auto",
    seed: int = 0,
    temperature: float | None = None,
    top_p: float = 1.0,
    stop_on_eos: bool = False,
    source: BundleSource | None = None,
) -> GenerationResult:
    """Verify a tokenizer-bound native bundle and generate one text continuation."""

    _validate_generation_arguments(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        stop_on_eos=stop_on_eos,
    )
    target_device = _resolve_device(device)
    target_dtype, dtype_name = _resolve_dtype(dtype)
    bundle = Path(bundle_directory)
    report = verify_deepseek_v4_pretrained_bundle(bundle)
    if not report.is_complete:
        raise ValueError("native bundle verification failed: " + "; ".join(report.errors))
    if not report.generation_ready:
        raise ValueError(
            "native bundle has no checksum-bound byte-v1 tokenizer; generation requires "
            "a format_version 2 bundle."
        )
    tokenizer = load_deepseek_v4_pretrained_tokenizer(bundle)
    config = DeepSeekV4Config.from_json_file(bundle / _CONFIG_NAME)
    prompt_ids = tokenizer.encode(prompt)
    prompt_token_count = int(prompt_ids.numel())
    if prompt_token_count + max_new_tokens > config.max_position_embeddings:
        raise ValueError(
            "prompt tokens plus max_new_tokens exceed config max_position_embeddings: "
            f"{prompt_token_count} + {max_new_tokens} > {config.max_position_embeddings}."
        )

    model = DeepSeekV4ForCausalLM.from_pretrained(
        bundle,
        device=target_device,
        dtype=target_dtype,
    ).eval()
    input_ids = prompt_ids.unsqueeze(0).to(target_device)
    generator = (
        torch.Generator(device=target_device).manual_seed(seed)
        if temperature is not None
        else None
    )
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=temperature is not None,
        temperature=1.0 if temperature is None else float(temperature),
        top_p=float(top_p),
        generator=generator,
        stop_on_eos=stop_on_eos,
    )
    continuation = generated[0, prompt_token_count:]
    continuation_token_ids = [int(token_id) for token_id in continuation.cpu().tolist()]
    resolved_source = source or BundleSource(kind="local", bundle_path=str(bundle))
    return GenerationResult(
        schema_version=1,
        source=resolved_source,
        bundle_manifest_sha256=report.manifest_sha256,
        config_sha256=report.config_sha256,
        tokenizer_sha256=report.tokenizer_sha256,
        device=str(target_device),
        dtype=dtype_name,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        seed=seed,
        temperature=None if temperature is None else float(temperature),
        top_p=float(top_p),
        stop_on_eos=stop_on_eos,
        prompt_text=prompt,
        prompt_token_count=prompt_token_count,
        max_new_tokens=max_new_tokens,
        generated_new_tokens=len(continuation_token_ids),
        continuation_token_ids=continuation_token_ids,
        generated_text=tokenizer.decode(generated[0]),
        continuation_text=tokenizer.decode(continuation),
    )


def _read_prompt_file(path: Path) -> str:
    if path.stat().st_size > _MAX_PROMPT_FILE_BYTES:
        raise ValueError("prompt file exceeds the 16 MiB safety limit.")
    return path.read_text(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text from a tokenizer-bound native nano-deepseek-v4 bundle.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path, help="Local native bundle directory.")
    source.add_argument("--hf-repo", help="Hugging Face model repository ID.")
    parser.add_argument("--revision", help="Required branch, tag, or commit for --hf-repo.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Optional Hub cache root; the verified regular-file bundle is "
            "materialized beneath it."
        ),
    )
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="UTF-8 generation prompt.")
    prompt.add_argument("--prompt-file", type=Path, help="Read the UTF-8 prompt from a file.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=("auto", "float32", "bfloat16", "float16"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        help="Enable seeded sampling at this temperature; omit for greedy generation.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable receipt.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.hf_repo is not None:
            if args.revision is None:
                raise ValueError("--revision is required with --hf-repo.")
        else:
            if args.revision is not None:
                raise ValueError("--revision can only be used with --hf-repo.")
            if args.cache_dir is not None:
                raise ValueError("--cache-dir can only be used with --hf-repo.")

        if args.prompt is not None:
            prompt = args.prompt
        elif args.prompt_file is not None:
            prompt = _read_prompt_file(args.prompt_file)
        elif not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            raise ValueError("provide --prompt, --prompt-file, or pipe a prompt on stdin.")

        _validate_generation_arguments(
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_on_eos=args.stop_on_eos,
        )
        _resolve_device(args.device)
        _resolve_dtype(args.dtype)

        if args.hf_repo is not None:
            assert args.revision is not None
            source = download_huggingface_bundle(
                args.hf_repo,
                args.revision,
                cache_dir=args.cache_dir,
            )
        else:
            source = BundleSource(kind="local", bundle_path=str(args.bundle))

        result = run_text_generation(
            source.bundle_path,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype=args.dtype,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_on_eos=args.stop_on_eos,
            source=source,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(result.generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
