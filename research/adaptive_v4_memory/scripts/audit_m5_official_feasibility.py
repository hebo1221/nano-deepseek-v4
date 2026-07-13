from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import torch

REPOSITORIES = {
    "flash": (
        "deepseek-ai/DeepSeek-V4-Flash",
        "60d8d70770c6776ff598c94bb586a859a38244f1",
    ),
    "pro": (
        "deepseek-ai/DeepSeek-V4-Pro",
        "b5968e9190ef611bbf34a7229255be88a0e937c1",
    ),
}


def _json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "nano-deepseek-v4-audit/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url}")
    return payload


def _sibling_size(sibling: dict[str, Any]) -> int:
    lfs = sibling.get("lfs")
    raw_size = lfs.get("size", 0) if isinstance(lfs, dict) else sibling.get("size", 0)
    if isinstance(raw_size, bool) or not isinstance(raw_size, int):
        raise ValueError(f"Invalid repository file size: {raw_size!r}")
    return raw_size


def _linux_memory() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            values[name] = int(raw.strip().split()[0]) * 1024
    if len(values) != 4:
        raise RuntimeError("Could not read required Linux memory counters.")
    return values


def _repo(name: str, revision: str) -> dict[str, Any]:
    quoted = urllib.parse.quote(name, safe="/")
    metadata = _json(
        f"https://huggingface.co/api/models/{quoted}?revision={revision}&blobs=true"
    )
    if metadata.get("sha") != revision:
        raise RuntimeError(f"Repository {name} did not resolve to pinned revision {revision}.")
    siblings = metadata.get("siblings", [])
    tensor_files = [
        sibling
        for sibling in siblings
        if isinstance(sibling, dict) and str(sibling.get("rfilename", "")).endswith(".safetensors")
    ]
    tensor_bytes = sum(_sibling_size(sibling) for sibling in tensor_files)
    config = _json(f"https://huggingface.co/{quoted}/resolve/{revision}/config.json")
    return {
        "repository": name,
        "revision": revision,
        "safetensors_files": len(tensor_files),
        "safetensors_bytes": tensor_bytes,
        "config": {
            key: config.get(key)
            for key in (
                "architectures",
                "model_type",
                "hidden_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "max_position_embeddings",
                "torch_dtype",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official V4 snapshot feasibility.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    memory = _linux_memory()
    disk = shutil.disk_usage(Path.cwd())
    host = {
        "physical_memory_bytes": memory["MemTotal"],
        "available_memory_bytes": memory["MemAvailable"],
        "swap_total_bytes": memory["SwapTotal"],
        "swap_free_bytes": memory["SwapFree"],
        "available_memory_plus_swap_bytes": memory["MemAvailable"] + memory["SwapFree"],
        "disk_free_bytes": disk.free,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "installed_runtimes": {
            name: importlib.util.find_spec(name) is not None
            for name in ("transformers", "vllm", "sglang", "flashinfer", "triton")
        },
    }
    repositories = {key: _repo(*value) for key, value in REPOSITORIES.items()}
    for repository in repositories.values():
        payload_bytes = repository["safetensors_bytes"]
        repository["host_feasibility"] = {
            "fits_available_memory_without_quantized_streaming": payload_bytes
            <= host["available_memory_plus_swap_bytes"],
            "fits_free_disk": payload_bytes <= host["disk_free_bytes"],
        }
    payload = {
        "schema_version": 1,
        "experiment_id": "m5-official-feasibility-audit-v1",
        "source": "Official Hugging Face repository API and pinned config.json revisions",
        "host": host,
        "repositories": repositories,
        "decision_rule": (
            "Do not download or claim an official-weight run unless the pinned tensor payload "
            "fits available memory and a supported official runtime is installed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
