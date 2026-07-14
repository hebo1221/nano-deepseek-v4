from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from p3_sequence_gate import require_p3_sequence_gate
from validate_p3_natural_suite_manifest import validate_manifest

DOWNLOADABLE_BENCHMARKS = ("SCBench", "LongBench-v2", "LongMemEval", "MRCR")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty, "implementation_sha256": sha256(Path(__file__))}


def _download_file(*, dataset: dict[str, Any], entry: dict[str, Any], output_root: Path) -> Path:
    local_dir = output_root / dataset["repo_id"].replace("/", "--")
    result = hf_hub_download(
        repo_id=dataset["repo_id"],
        filename=entry["path"],
        revision=dataset["revision"],
        repo_type="dataset",
        local_dir=local_dir,
    )
    path = Path(result)
    observed_bytes = path.stat().st_size
    if observed_bytes != entry["bytes"]:
        raise ValueError(
            f"Byte-size mismatch for {entry['path']}: {observed_bytes} != {entry['bytes']}."
        )
    observed_digest = sha256(path)
    if observed_digest != entry["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {entry['path']}.")
    return path


def _json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON array of objects at {path}.")
    return payload


def _inspect_file(benchmark: str, path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        result["columns"] = parquet.schema_arrow.names
    else:
        payload = _json_rows(path)
        rows = len(payload)
        if benchmark == "LongBench-v2":
            identifiers = [row.get("_id") for row in payload]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("LongBench v2 contains duplicate example IDs.")
            answers = [row.get("answer") for row in payload]
            if any(not isinstance(answer, str) or answer not in "ABCD" for answer in answers):
                raise ValueError("LongBench v2 contains an invalid answer label.")
        elif benchmark == "LongMemEval":
            identifiers = [row.get("question_id") for row in payload]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("LongMemEval contains duplicate question IDs.")
            required = {"question_id", "question_type", "question", "answer", "haystack_sessions"}
            if any(not required.issubset(row) for row in payload):
                raise ValueError("LongMemEval is missing required fields.")
    if rows != entry["rows"]:
        raise ValueError(f"Row mismatch for {entry['path']}: {rows} != {entry['rows']}.")
    result["rows"] = rows
    return result


def _validate_scbench_turns(files: list[Path], expected: dict[str, Any]) -> dict[str, int]:
    contexts = 0
    turns = 0
    for path in files:
        task = path.parent.name
        table = pq.read_table(path, columns=["multi_turns"])
        task_turns = sum(len(value) for value in table.column("multi_turns").to_pylist())
        task_rows = table.num_rows
        contract = expected[task]
        if task_rows != contract["rows"] or task_turns != contract["turns"]:
            raise ValueError(f"SCBench row/turn drift for {task}.")
        contexts += task_rows
        turns += task_turns
    if contexts != 922 or turns != 5143:
        raise ValueError("SCBench aggregate row/turn totals drifted.")
    return {"shared_contexts": contexts, "turns_per_mode": turns}


def prepare(
    *, manifest: dict[str, Any], output_root: Path, selected: tuple[str, ...]
) -> dict[str, Any]:
    inventories: dict[str, Any] = {}
    for benchmark in selected:
        contract = manifest["benchmarks"][benchmark]
        dataset = contract["dataset"]
        local_files: list[Path] = []
        file_inventory: list[dict[str, Any]] = []
        for entry in dataset["files"]:
            path = _download_file(dataset=dataset, entry=entry, output_root=output_root)
            local_files.append(path)
            file_inventory.append(_inspect_file(benchmark, path, entry))
        inventory: dict[str, Any] = {
            "repo_id": dataset["repo_id"],
            "revision": dataset["revision"],
            "license": dataset["license"],
            "files": file_inventory,
        }
        if benchmark == "SCBench":
            inventory["observed"] = _validate_scbench_turns(local_files, contract["tasks"])
        inventories[benchmark] = inventory
    return inventories


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and verify frozen P3 natural datasets.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-data"),
    )
    parser.add_argument("--benchmark", action="append", choices=DOWNLOADABLE_BENCHMARKS)
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
    decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source = source_state()
    if source["dirty"]:
        raise RuntimeError("Natural dataset acquisition requires a clean experiment source tree.")

    selected = tuple(args.benchmark or DOWNLOADABLE_BENCHMARKS)
    inventories = prepare(
        manifest=manifest, output_root=args.output_root.resolve(), selected=selected
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "p3-natural-dataset-inventory-v1",
        "source": source,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": hashlib.sha256(raw_manifest).hexdigest(),
            "validation": validation,
        },
        "sequence_gate": decision,
        "benchmarks": inventories,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    target = args.output_root / "dataset-inventory.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    print(json.dumps({"inventory": str(target), "benchmarks": list(selected)}, indent=2))


if __name__ == "__main__":
    main()
