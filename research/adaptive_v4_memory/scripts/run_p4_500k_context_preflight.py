from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run_p4_systems_matrix as systems
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

SCALES = systems.SCALES
POLICIES = systems.POLICIES
CONTEXT = 500_000
GENERATION = 128
BATCH = 1
ACTIVE_REQUESTS = 1
CELL_TIMEOUT_SECONDS = systems.CELL_TIMEOUT_SECONDS
EXPECTED_CELLS = len(SCALES)
IMPLEMENTATION_PATHS = (
    *systems.IMPLEMENTATION_PATHS,
    "research/adaptive_v4_memory/manifests/p4-500k-context-preflight-v1.json",
    "research/adaptive_v4_memory/scripts/run_p4_500k_context_preflight.py",
    "research/adaptive_v4_memory/scripts/summarize_p4_500k_context_preflight.py",
)


def implementation_digest() -> str:
    tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = {line.split("\t", 1)[1] for line in tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in IMPLEMENTATION_PATHS
        if path not in paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in paths)
    ]
    if missing:
        raise RuntimeError(f"Untracked P4 500K implementation paths: {missing}")
    return hashlib.sha256(tree.encode()).hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _status(policy_attempts: dict[str, dict[str, Any]]) -> str:
    successes = sum(row.get("status") == "success" for row in policy_attempts.values())
    return "complete" if successes == len(POLICIES) else "partial" if successes else "failed"


def _failure_status(error: Exception) -> str:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "oom"
    if isinstance(error, TimeoutError):
        return "timeout"
    return "error"


def _artifact_valid(
    path: Path,
    *,
    scale: str,
    digest: str,
    manifest_digest: str,
    p3_digest: str,
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    attempts = payload.get("policy_attempts", {})
    if not (
        payload.get("experiment_id") == "p4-500k-context-preflight-cell-v1"
        and payload.get("scale") == scale
        and payload.get("context_tokens") == CONTEXT
        and payload.get("generation_tokens") == GENERATION
        and payload.get("batch") == BATCH
        and payload.get("active_requests") == ACTIVE_REQUESTS
        and type(payload.get("cell_timeout_seconds")) in (int, float)
        and 0.0 < payload["cell_timeout_seconds"] <= CELL_TIMEOUT_SECONDS
        and set(attempts) == set(POLICIES)
        and all(
            row.get("status") in {"success", "oom", "timeout", "error"} for row in attempts.values()
        )
        and payload.get("status") == _status(attempts)
        and payload.get("source", {}).get("implementation_digest") == digest
        and payload.get("manifest", {}).get("sha256") == manifest_digest
        and payload.get("p3_audit", {}).get("sha256") == p3_digest
    ):
        return False
    successful = [row for row in attempts.values() if row["status"] == "success"]
    return all(
        row.get("run", {}).get("context_tokens") == CONTEXT
        and row.get("run", {}).get("generation_tokens") == GENERATION
        and row.get("run", {}).get("input_digest") == payload.get("input_digest")
        and isinstance(row.get("run", {}).get("prediction_digest"), str)
        for row in successful
    )


def _matrix_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "scale": payload["scale"],
        "status": payload["status"],
        "policy_status": {
            policy: payload["policy_attempts"][policy]["status"] for policy in POLICIES
        },
        "artifact": {"path": str(path), "sha256": systems.sha256(path)},
    }


def _write_matrix(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    implementation: str,
    manifest: Path,
    p3_audit: Path,
) -> None:
    rows.sort(key=lambda row: row["scale"])
    _write_json(
        path,
        {
            "schema_version": 1,
            "experiment_id": "p4-500k-context-preflight-progress-v1",
            "source_commit": _head(),
            "implementation_digest": implementation,
            "manifest": {"path": str(manifest), "sha256": systems.sha256(manifest)},
            "p3_audit": {"path": str(p3_audit), "sha256": systems.sha256(p3_audit)},
            "expected_cells": EXPECTED_CELLS,
            "terminal_cells": len(rows),
            "successful_policy_attempts": sum(
                status == "success" for row in rows for status in row["policy_status"].values()
            ),
            "terminal_policy_attempts": len(rows) * len(POLICIES),
            "runs": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen P4 500K context preflight.")
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument(
        "--cell-timeout-seconds", type=float, default=CELL_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p4-500k-context-preflight-v1.json"),
    )
    parser.add_argument(
        "--p3-audit",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-suite.summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/500k-context-preflight"),
    )
    parser.add_argument(
        "--matrix-progress",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p4/500k-context-preflight-matrix.json"
        ),
    )
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells <= 0:
        raise ValueError("max-new-cells must be positive.")
    systems.validate_cell_timeout(args.cell_timeout_seconds)
    if _dirty():
        raise RuntimeError("P4 500K execution requires a clean source tree.")
    systems.require_p3_audit(args.p3_audit)
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("experiment_id") != "p4-500k-context-preflight-v1"
        or manifest.get("context_tokens") != CONTEXT
        or manifest.get("expected_scale_cells") != EXPECTED_CELLS
        or manifest.get("expected_policy_attempts") != EXPECTED_CELLS * len(POLICIES)
        or manifest.get("execution", {}).get("maximum_cell_timeout_seconds")
        != CELL_TIMEOUT_SECONDS
    ):
        raise RuntimeError("The frozen P4 500K preflight manifest is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("P4 500K execution requires CUDA.")
    lock = acquire_gpu_lock("p4-500k-context-preflight")
    implementation = implementation_digest()
    manifest_digest = systems.sha256(args.manifest)
    p3_digest = systems.sha256(args.p3_audit)
    selected = [scale for scale in SCALES if not args.scale or scale in args.scale]
    rows: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        path = args.output_root / scale / "cell.json"
        if _artifact_valid(
            path,
            scale=scale,
            digest=implementation,
            manifest_digest=manifest_digest,
            p3_digest=p3_digest,
        ):
            rows[scale] = _matrix_row(path, json.loads(path.read_text()))
    new_cells = 0
    device = torch.device("cuda")
    for scale in selected:
        path = args.output_root / scale / "cell.json"
        if scale in rows:
            continue
        if args.max_new_cells is not None and new_cells >= args.max_new_cells:
            break
        started = time.monotonic()
        attempts: dict[str, dict[str, Any]] = {}
        input_digest: str | None = None
        previous_alarm_handler = systems.arm_cell_timeout(args.cell_timeout_seconds)
        try:
            checkpoint = args.training_root / scale / "seed-6071401" / f"{scale}-step-1000.pt"
            model = systems._load_model(checkpoint, device)
            prompts, decode, input_digest = systems.generate_inputs(
                model,
                context=CONTEXT,
                generation=GENERATION,
                batch=BATCH,
                active_requests=ACTIVE_REQUESTS,
                seed=9_071_500,
            )
            order = POLICIES if scale == SCALES[0] else tuple(reversed(POLICIES))
            for policy in order:
                if time.monotonic() - started > args.cell_timeout_seconds:
                    attempts[policy] = {
                        "status": "timeout",
                        "error_type": "TimeoutError",
                        "error": "P4 500K scale cell exceeded its frozen wall-time limit.",
                    }
                    continue
                try:
                    run = systems.run_policy(
                        model,
                        policy=policy,
                        prompts=prompts,
                        decode_tokens=decode,
                        input_digest=input_digest,
                        device=device,
                    )
                    attempts[policy] = {"status": "success", "run": run}
                except Exception as error:
                    attempts[policy] = {
                        "status": _failure_status(error),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    systems._cleanup()
            del prompts, decode, model
            systems._cleanup()
        except Exception as error:
            outcome = _failure_status(error)
            for policy in POLICIES:
                attempts.setdefault(
                    policy,
                    {"status": outcome, "error_type": type(error).__name__, "error": str(error)},
                )
            systems._cleanup()
        finally:
            systems.cancel_cell_timeout(previous_alarm_handler)
        payload = {
            "schema_version": 1,
            "experiment_id": "p4-500k-context-preflight-cell-v1",
            "status": _status(attempts),
            "scale": scale,
            "context_tokens": CONTEXT,
            "generation_tokens": GENERATION,
            "batch": BATCH,
            "active_requests": ACTIVE_REQUESTS,
            "cell_timeout_seconds": args.cell_timeout_seconds,
            "input_digest": input_digest,
            "policy_attempts": attempts,
            "elapsed_seconds": time.monotonic() - started,
            "source": {
                "commit": _head(),
                "dirty": False,
                "implementation_digest": implementation,
            },
            "manifest": {"path": str(args.manifest), "sha256": manifest_digest},
            "p3_audit": {"path": str(args.p3_audit), "sha256": p3_digest},
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(device),
            },
            "command": [sys.executable, *sys.argv],
            "claim_boundary": (
                "One feasibility attempt per policy; no performance claim. The POSIX timer "
                "does not prove preemption of an uninterruptible native CUDA call."
            ),
        }
        _write_json(path, payload)
        rows[scale] = _matrix_row(path, payload)
        new_cells += 1
        _write_matrix(
            args.matrix_progress,
            rows=list(rows.values()),
            implementation=implementation,
            manifest=args.manifest,
            p3_audit=args.p3_audit,
        )
    _write_matrix(
        args.matrix_progress,
        rows=list(rows.values()),
        implementation=implementation,
        manifest=args.manifest,
        p3_audit=args.p3_audit,
    )
    del lock
    print(json.dumps({"terminal_cells": len(rows), "new_cells": new_cells}, sort_keys=True))


if __name__ == "__main__":
    main()
