from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest_set(files: dict[str, str]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return payload


def verify_snapshot(snapshot: Path, model: dict[str, Any]) -> dict[str, Any]:
    if not snapshot.is_dir():
        raise ValueError(f"Model snapshot directory does not exist: {snapshot}")

    expected = model["snapshot_files_sha256"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Frozen model snapshot file map is empty or invalid.")
    expected = {str(name): str(digest) for name, digest in expected.items()}
    expected_set_digest = canonical_digest_set(expected)
    if expected_set_digest != model["snapshot_digest_set_sha256"]:
        raise ValueError("Frozen model snapshot digest-set is internally inconsistent.")

    actual_names = {path.name for path in snapshot.iterdir() if path.is_file()}
    expected_names = set(expected)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise ValueError(f"Model snapshot file set drifted; missing={missing}, extra={extra}.")

    files: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = snapshot / name
        observed_digest = sha256(path)
        if observed_digest != expected[name]:
            raise ValueError(f"Model snapshot SHA-256 drifted: {name}.")
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": observed_digest,
            }
        )

    index = _load_object(snapshot / "model.safetensors.index.json", "Weight index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Weight index has no non-empty weight_map.")
    indexed_shards = {str(name) for name in weight_map.values()}
    frozen_shards = {
        name
        for name in expected
        if name.startswith("model-") and name.endswith(".safetensors")
    }
    if indexed_shards != frozen_shards:
        raise ValueError("Weight index shard set does not match the frozen snapshot.")
    snapshot_bytes = sum(row["bytes"] for row in files)
    if snapshot_bytes != model["snapshot_bytes"]:
        raise ValueError(
            f"Snapshot byte total drifted: {snapshot_bytes} != {model['snapshot_bytes']}."
        )
    weight_shard_file_bytes = sum((snapshot / name).stat().st_size for name in indexed_shards)
    if weight_shard_file_bytes != model["weight_shard_file_bytes"]:
        raise ValueError(
            "Weight-shard file byte total drifted: "
            f"{weight_shard_file_bytes} != {model['weight_shard_file_bytes']}."
        )
    metadata = index.get("metadata")
    if not isinstance(metadata, dict) or "total_size" not in metadata:
        raise ValueError("Weight index metadata has no total_size.")
    indexed_tensor_bytes = int(metadata["total_size"])
    if indexed_tensor_bytes != model["indexed_tensor_bytes"]:
        raise ValueError(
            "Indexed tensor byte total drifted: "
            f"{indexed_tensor_bytes} != {model['indexed_tensor_bytes']}."
        )

    config = _load_object(snapshot / "config.json", "Model config")
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or model["architecture"] not in architectures:
        raise ValueError("Model config architecture does not match the frozen contract.")
    maximum_context = int(config.get("max_position_embeddings", 0))
    if maximum_context != model["maximum_supported_context_tokens"]:
        raise ValueError("Model config maximum context does not match the frozen contract.")

    return {
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_digest_set_sha256": expected_set_digest,
        "file_count": len(files),
        "snapshot_bytes": snapshot_bytes,
        "weight_shard_file_bytes": weight_shard_file_bytes,
        "indexed_tensor_bytes": indexed_tensor_bytes,
        "indexed_tensor_count": len(weight_map),
        "architecture": model["architecture"],
        "maximum_supported_context_tokens": maximum_context,
        "files": files,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen P3 natural-model snapshot.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/assets/models/"
            "cdbee75f17c01a7cc42f958dc650907174af0554"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/"
            "natural-model-snapshot-verification.json"
        ),
    )
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    result = verify_snapshot(args.snapshot, manifest["model"])
    payload = {
        "schema_version": 1,
        "experiment_id": "p3-natural-model-snapshot-verification-v1",
        "status": "verified",
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "model": result,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "status": "verified"}, indent=2))


if __name__ == "__main__":
    main()
