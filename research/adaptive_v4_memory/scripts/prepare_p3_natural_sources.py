from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from p3_sequence_gate import require_p3_sequence_gate
from validate_p3_natural_suite_manifest import validate_manifest

BENCHMARKS = ("SCBench", "LongBench-v2", "LongMemEval")
PREFETCH_SCOPE = {
    "immutable_public_code_only": True,
    "benchmark_dataset_payload_acquired": False,
    "baseline_selected": False,
    "dataset_generated": False,
    "model_inference_performed": False,
}
PREFETCH_AMENDMENT = (
    "permit prefetch and cryptographic verification only for the already preregistered "
    "immutable model revision and pinned public code dependencies before the P2 causal "
    "gate; retain the gate for benchmark dataset payload acquisition, baseline selection, "
    "dataset generation, and every prediction"
)


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
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ValueError(f"Refusing to reuse a partial source checkout: {temporary}")
    temporary.mkdir()
    try:
        _git("init", "--quiet", cwd=temporary)
        _git("remote", "add", "origin", contract["repository"], cwd=temporary)
        _git("fetch", "--quiet", "--depth", "1", "origin", contract["revision"], cwd=temporary)
        _git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def acquire_or_verify_source(destination: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Resume from an exact pinned checkout, but never tolerate source drift."""

    if not destination.exists():
        acquire_source(destination, contract)
    return verify_source(destination, contract)


def require_prefetch_authorization(manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [
        amendment
        for amendment in manifest.get("amendments", [])
        if amendment.get("date") == "2026-07-14"
        and amendment.get("change") == PREFETCH_AMENDMENT
    ]
    if len(matches) != 1:
        raise ValueError("Natural-source prefetch amendment is missing or ambiguous.")
    amendment = matches[0]
    encoded = json.dumps(amendment, sort_keys=True, separators=(",", ":")).encode()
    return {
        "date": amendment["date"],
        "change": amendment["change"],
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def verify_source(source: Path, contract: dict[str, Any]) -> dict[str, Any]:
    observed_revision = _git("rev-parse", "HEAD", cwd=source)
    if observed_revision != contract["revision"]:
        raise ValueError(f"Source revision drifted: {observed_revision} != {contract['revision']}.")
    if _git("status", "--porcelain", cwd=source):
        raise ValueError("Pinned source checkout contains tracked or untracked modifications.")

    files: list[dict[str, Any]] = []
    expected_files = contract["files_sha256"]
    for relative, expected_digest in sorted(expected_files.items()):
        path = source / relative
        if not path.is_file():
            raise ValueError(f"Pinned source file is missing: {relative}.")
        observed_digest = sha256(path)
        if observed_digest != expected_digest:
            raise ValueError(f"Pinned source SHA-256 drifted: {relative}.")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": observed_digest})

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
        "clean_untracked_tree": True,
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


def workspace_source() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"commit": commit, "dirty": dirty}


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
    parser.add_argument(
        "--prefetch-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/assets/sources/natural"
        ),
    )
    parser.add_argument(
        "--prefetch-only",
        action="store_true",
        help=(
            "Acquire and verify only immutable public code dependencies under the "
            "preregistered pre-gate exception."
        ),
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
    prefetch_authorization = require_prefetch_authorization(manifest)
    source = workspace_source()
    if source["dirty"]:
        raise RuntimeError("Natural source acquisition requires a clean experiment source tree.")
    selected = tuple(args.benchmark or BENCHMARKS)

    inventories: dict[str, Any] = {}
    for benchmark in selected:
        contract = manifest["benchmarks"][benchmark]["upstream_code"]
        relative = benchmark.lower().replace("-", "_")
        prefetched = args.prefetch_root / relative
        if args.prefetch_only or prefetched.exists():
            destination = prefetched
        else:
            destination = args.output_root / relative
        inventories[benchmark] = acquire_or_verify_source(destination, contract)

    if args.prefetch_only:
        payload = {
            "schema_version": 1,
            "experiment_id": "p3-natural-source-prefetch-v1",
            "status": "verified",
            "source": source,
            "manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "validation": validation,
            },
            "preregistered_exception": {
                "manifest_amendment": prefetch_authorization,
                "scope": PREFETCH_SCOPE,
            },
            "implementation_sha256": sha256(Path(__file__)),
            "benchmarks": inventories,
        }
        output = args.prefetch_root / "prefetch-inventory.json"
        atomic_json(output, payload)
        print(
            json.dumps(
                {
                    "benchmarks": list(selected),
                    "status": "verified",
                    "scope": "immutable-public-code-only",
                    "output": str(output),
                },
                indent=2,
            )
        )
        return

    sequence_gate = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)

    payload = {
        "schema_version": 1,
        "experiment_id": "p3-natural-source-inventory-v1",
        "status": "verified",
        "source": source,
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "validation": validation,
        },
        "sequence_gate": sequence_gate,
        "prefetch_reuse": {
            "root": str(args.prefetch_root.resolve()),
            "manifest_amendment": prefetch_authorization,
            "verified_again_after_sequence_gate": True,
        },
        "implementation_sha256": sha256(Path(__file__)),
        "benchmarks": inventories,
    }
    atomic_json(args.output_root / "source-inventory.json", payload)
    print(json.dumps({"benchmarks": list(selected), "status": "verified"}, indent=2))


if __name__ == "__main__":
    main()
