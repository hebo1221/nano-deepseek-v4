from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from p3_sequence_gate import require_p3_sequence_gate
from validate_p3_natural_suite_manifest import validate_manifest

BENCHMARKS = ("SCBench", "LongBench-v2", "LongMemEval")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def acquire_source(destination: Path, contract: dict[str, Any]) -> None:
    if destination.exists():
        raise ValueError(f"Refusing to reuse an existing source directory: {destination}")
    destination.mkdir(parents=True)
    _git("init", "--quiet", cwd=destination)
    _git("remote", "add", "origin", contract["repository"], cwd=destination)
    _git("fetch", "--quiet", "--depth", "1", "origin", contract["revision"], cwd=destination)
    _git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=destination)


def verify_source(source: Path, contract: dict[str, Any]) -> dict[str, Any]:
    observed_revision = _git("rev-parse", "HEAD", cwd=source)
    if observed_revision != contract["revision"]:
        raise ValueError(
            f"Source revision drifted: {observed_revision} != {contract['revision']}."
        )
    if _git("status", "--porcelain", "--untracked-files=no", cwd=source):
        raise ValueError("Pinned source checkout contains tracked modifications.")

    files: list[dict[str, Any]] = []
    expected_files = contract["files_sha256"]
    for relative, expected_digest in sorted(expected_files.items()):
        path = source / relative
        if not path.is_file():
            raise ValueError(f"Pinned source file is missing: {relative}.")
        observed_digest = sha256(path)
        if observed_digest != expected_digest:
            raise ValueError(f"Pinned source SHA-256 drifted: {relative}.")
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": observed_digest}
        )

    license_path = source / "LICENSE"
    if not license_path.is_file():
        raise ValueError("Pinned source checkout has no root LICENSE file.")
    license_digest = sha256(license_path)
    if license_digest != contract["license_sha256"]:
        raise ValueError("Pinned source LICENSE SHA-256 drifted.")

    return {
        "repository": contract["repository"],
        "revision": observed_revision,
        "path": str(source.resolve()),
        "clean_tracked_tree": True,
        "license": contract["license"],
        "license_path": "LICENSE",
        "license_sha256": license_digest,
        "files": files,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire and verify frozen P3 natural evaluation sources."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-sources"),
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

    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    validation = validate_manifest(manifest)
    sequence_gate = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    selected = tuple(args.benchmark or BENCHMARKS)

    inventories: dict[str, Any] = {}
    for benchmark in selected:
        contract = manifest["benchmarks"][benchmark]["upstream_code"]
        destination = args.output_root / benchmark.lower().replace("-", "_")
        acquire_source(destination, contract)
        inventories[benchmark] = verify_source(destination, contract)

    payload = {
        "schema_version": 1,
        "experiment_id": "p3-natural-source-inventory-v1",
        "status": "verified",
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "validation": validation,
        },
        "sequence_gate": sequence_gate,
        "implementation_sha256": sha256(Path(__file__)),
        "benchmarks": inventories,
    }
    atomic_json(args.output_root / "source-inventory.json", payload)
    print(json.dumps({"benchmarks": list(selected), "status": "verified"}, indent=2))


if __name__ == "__main__":
    main()
