from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from p3_sequence_gate import require_p3_sequence_gate
from validate_p3_natural_safety_manifest import validate_manifest

BENCHMARKS = ("LongSafety", "IFEval")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    _require(path.is_file(), f"Frozen asset is missing: {path}.")
    _require(path.stat().st_size == entry["bytes"], f"Frozen asset byte size drifted: {path}.")
    observed = sha256(path)
    _require(observed == entry["sha256"], f"Frozen asset SHA-256 drifted: {path}.")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def validate_longsafety_rows(
    rows: Any, *, expected_rows: int, required_fields: list[str]
) -> dict[str, Any]:
    _require(isinstance(rows, list) and len(rows) == expected_rows, "LongSafety row count drifted.")
    identifiers: set[int] = set()
    safety_types: set[str] = set()
    task_types: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict) and set(row) == set(required_fields), "LongSafety schema drifted.")
        identifier = row.get("id")
        _require(
            isinstance(identifier, int) and identifier not in identifiers,
            "LongSafety id is invalid or duplicated.",
        )
        identifiers.add(identifier)
        _require(
            isinstance(row.get("link"), list)
            and all(isinstance(value, str) for value in row["link"])
            and isinstance(row.get("key_words"), list)
            and all(isinstance(value, str) for value in row["key_words"])
            and isinstance(row.get("length"), int)
            and row["length"] > 0
            and isinstance(row.get("doc_num"), int)
            and row["doc_num"] > 0
            and all(
                isinstance(row.get(field), str) and bool(row[field].strip())
                for field in ("safety_type", "instruction", "task_type", "context")
            ),
            f"LongSafety row fields drifted: {identifier}.",
        )
        safety_types.add(row["safety_type"])
        task_types.add(row["task_type"])
    return {
        "rows": len(rows),
        "unique_ids": len(identifiers),
        "safety_types": sorted(safety_types),
        "task_types": sorted(task_types),
    }


def validate_ifeval_rows(
    rows: list[Any], *, expected_rows: int, required_fields: list[str]
) -> dict[str, Any]:
    _require(len(rows) == expected_rows, "IFEval row count drifted.")
    keys: set[int] = set()
    instruction_ids: set[str] = set()
    for row in rows:
        _require(isinstance(row, dict) and set(row) == set(required_fields), "IFEval schema drifted.")
        key = row.get("key")
        ids, kwargs = row.get("instruction_id_list"), row.get("kwargs")
        _require(
            isinstance(key, int)
            and key not in keys
            and isinstance(row.get("prompt"), str)
            and bool(row["prompt"].strip())
            and isinstance(ids, list)
            and bool(ids)
            and all(isinstance(value, str) and value for value in ids)
            and isinstance(kwargs, list)
            and len(kwargs) == len(ids)
            and all(isinstance(value, dict) for value in kwargs),
            f"IFEval row fields drifted: {key}.",
        )
        keys.add(key)
        instruction_ids.update(ids)
    return {
        "rows": len(rows),
        "unique_keys": len(keys),
        "unique_instruction_ids": len(instruction_ids),
    }


def _dataset_file(
    *, contract: dict[str, Any], entry: dict[str, Any], output_root: Path
) -> Path:
    local_dir = output_root / "datasets" / contract["repo_id"].replace("/", "--")
    path = Path(
        hf_hub_download(
            repo_id=contract["repo_id"],
            filename=entry["path"],
            revision=contract["revision"],
            repo_type="dataset",
            local_dir=local_dir,
        )
    )
    verify_file(path, entry)
    return path


def _source_file(
    *, benchmark: str, contract: dict[str, Any], entry: dict[str, Any], output_root: Path
) -> Path:
    destination = output_root / "sources" / benchmark.lower() / entry["path"]
    if destination.exists():
        verify_file(destination, entry)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    request = urllib.request.Request(
        f"{contract['raw_base']}/{entry['path']}",
        headers={"User-Agent": "adaptive-v4-memory-repro/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    verify_file(temporary, entry)
    temporary.replace(destination)
    return destination


def _jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prepare_assets(
    *, manifest: dict[str, Any], output_root: Path, selected: tuple[str, ...]
) -> dict[str, Any]:
    inventory: dict[str, Any] = {}
    for benchmark in selected:
        contract = manifest["benchmarks"][benchmark]
        dataset_files: list[dict[str, Any]] = []
        data_path: Path | None = None
        for entry in contract["dataset"]["files"]:
            path = _dataset_file(
                contract=contract["dataset"], entry=entry, output_root=output_root
            )
            dataset_files.append(verify_file(path, entry))
            if "rows" in entry:
                data_path = path
        _require(data_path is not None, f"{benchmark} has no row-bearing dataset file.")
        assert data_path is not None
        if benchmark == "LongSafety":
            observed = validate_longsafety_rows(
                json.loads(data_path.read_text()),
                expected_rows=contract["prompt_protocol"]["expected_rows"],
                required_fields=contract["dataset"]["required_fields"],
            )
        else:
            observed = validate_ifeval_rows(
                _jsonl(data_path),
                expected_rows=contract["protocol"]["expected_prompts_per_arm"],
                required_fields=contract["dataset"]["required_fields"],
            )
        source_files = [
            verify_file(
                _source_file(
                    benchmark=benchmark,
                    contract=contract["upstream_code"],
                    entry=entry,
                    output_root=output_root,
                ),
                entry,
            )
            for entry in contract["upstream_code"]["files"]
        ]
        inventory[benchmark] = {
            "dataset": {
                "repo_id": contract["dataset"]["repo_id"],
                "revision": contract["dataset"]["revision"],
                "license": contract["dataset"]["license"],
                "files": dataset_files,
                "observed": observed,
            },
            "upstream_code": {
                "repository": contract["upstream_code"]["repository"],
                "revision": contract["upstream_code"]["revision"],
                "license": contract["upstream_code"]["license"],
                "files": source_files,
            },
        }
    return inventory


def _source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty, "implementation_sha256": sha256(Path(__file__))}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire frozen P3 natural safety assets.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/assets"),
    )
    parser.add_argument("--benchmark", action="append", choices=BENCHMARKS)
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--causal-gate",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"),
    )
    args = parser.parse_args()
    raw_manifest = args.manifest.read_bytes()
    manifest = json.loads(raw_manifest)
    validation = validate_manifest(manifest)
    sequence_gate = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source = _source_state()
    _require(not source["dirty"], "Natural safety acquisition requires a clean source tree.")
    selected = tuple(args.benchmark or BENCHMARKS)
    inventory = prepare_assets(
        manifest=manifest, output_root=args.output_root.resolve(), selected=selected
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "p3-natural-safety-asset-inventory-v1",
        "status": "verified",
        "source": source,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "validation": validation,
        },
        "sequence_gate": sequence_gate,
        "dataset_viewer_dependency": False,
        "benchmarks": inventory,
    }
    target = args.output_root / "inventory.json"
    _atomic_json(target, payload)
    print(json.dumps({"inventory": str(target), "benchmarks": list(selected)}, sort_keys=True))


if __name__ == "__main__":
    main()
