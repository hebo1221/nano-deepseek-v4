#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BLOCKER = Path(
    "research/adaptive_v4_memory/manifests/p4-production-resource-blocker-v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_spec(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != "p4-production-adapter-spec-v1":
        raise ValueError("The frozen P4 production adapter spec is required.")
    cell = payload.get("cell")
    required = ("scale", "context", "generation", "profile", "batch", "concurrency")
    if not isinstance(cell, dict) or any(key not in cell for key in required):
        raise ValueError("The P4 production adapter cell is incomplete.")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail closed when no production serving adapter is configured."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = validate_spec(args.spec)
    blocker = json.loads(BLOCKER.read_text())
    if blocker.get("experiment_id") != "p4-production-resource-blocker-v1":
        raise ValueError("The frozen production resource blocker is required.")
    message = {
        "failure_type": "external-fused-dynamic-production-runtime-unavailable",
        "cell": spec["cell"],
        "blocker": {"path": str(BLOCKER), "sha256": sha256(BLOCKER)},
        "output_written": False,
    }
    print(json.dumps(message, sort_keys=True), file=sys.stderr)
    raise SystemExit(78)


if __name__ == "__main__":
    main()
