from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import audit_p2_continuous_rank as audit
import p2_continuous_rank_contract as contract
from adaptive_v4_gpu_lock import acquire_gpu_lock


def _load_existing_audit(path: Path, experiment_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    contract.reject_supervision_fields(payload)
    contract.validate_payload_digest(payload)
    if payload.get("experiment_id") != experiment_id or payload.get("status") != "terminal":
        raise ValueError(f"Existing continuous-rank audit is invalid: {path}")
    return payload


def _validate_existing_cell(
    path: Path,
    *,
    experiment_id: str,
    scale: str,
    training_seed: int,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return audit._validate_cell(  # noqa: SLF001 - shared fail-closed schema validator
        path,
        experiment_id=experiment_id,
        scale=scale,
        training_seed=training_seed,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _run_worker(
    script: Path,
    *,
    manifest_path: Path,
    scale: str,
    training_seed: int,
    output: Path,
    full_forward_audit: Path | None = None,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--manifest",
        str(manifest_path),
        "--scale",
        scale,
        "--training-seed",
        str(training_seed),
        "--output",
        str(output),
        "--borrowed-gpu-lock",
    ]
    if full_forward_audit is not None:
        command.extend(("--full-forward-audit", str(full_forward_audit)))
    subprocess.run(command, check=True)


def _publish_or_validate_audit(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    existing = _load_existing_audit(path, str(payload["experiment_id"]))
    if existing is not None:
        if existing != payload:
            raise ValueError(f"Existing continuous-rank audit differs from recomputation: {path}")
        return existing
    contract.write_json_exclusive(path, payload)
    return payload


def run_full_forward(manifest_path: Path) -> dict[str, Any]:
    manifest = contract.load_manifest(manifest_path)
    worker = Path(__file__).with_name("collect_p2_full_forward_continuous_rank.py")
    lock = acquire_gpu_lock("p2-707-continuous-rank-full-forward")
    try:
        for scale in contract.SCALES:
            for training_seed in contract.TRAINING_SEEDS:
                output = contract.full_forward_output_path(scale, training_seed)
                payload = _validate_existing_cell(
                    output,
                    experiment_id=contract.FULL_FORWARD_EXPERIMENT_ID,
                    scale=scale,
                    training_seed=training_seed,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )
                if payload is None:
                    _run_worker(
                        worker,
                        manifest_path=manifest_path,
                        scale=scale,
                        training_seed=training_seed,
                        output=output,
                    )
                    payload = _validate_existing_cell(
                        output,
                        experiment_id=contract.FULL_FORWARD_EXPERIMENT_ID,
                        scale=scale,
                        training_seed=training_seed,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                    if payload is None:
                        raise RuntimeError(f"Full-forward worker did not publish {output}.")
    finally:
        lock.close()
    summary = audit.summarize_full_forward(manifest_path=manifest_path)
    return _publish_or_validate_audit(contract.full_forward_audit_path(), summary)


def run_exact_path(manifest_path: Path, full_audit_path: Path) -> dict[str, Any]:
    manifest = contract.load_manifest(manifest_path)
    phase_1 = audit._validate_full_audit(  # noqa: SLF001 - matrix prerequisite validator
        full_audit_path, manifest_path
    )
    if phase_1.get("exact_path_phase_permitted") is not True:
        raise RuntimeError("Full-forward continuous-rank gate did not permit exact-path work.")
    worker = Path(__file__).with_name("collect_p2_exact_path_layer_signals.py")
    lock = acquire_gpu_lock("p2-707-continuous-rank-exact-path")
    try:
        for scale in contract.SCALES:
            for training_seed in contract.TRAINING_SEEDS:
                output = contract.exact_path_output_path(scale, training_seed)
                payload = _validate_existing_cell(
                    output,
                    experiment_id=contract.EXACT_PATH_EXPERIMENT_ID,
                    scale=scale,
                    training_seed=training_seed,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )
                if payload is None:
                    _run_worker(
                        worker,
                        manifest_path=manifest_path,
                        scale=scale,
                        training_seed=training_seed,
                        output=output,
                        full_forward_audit=full_audit_path,
                    )
                    payload = _validate_existing_cell(
                        output,
                        experiment_id=contract.EXACT_PATH_EXPERIMENT_ID,
                        scale=scale,
                        training_seed=training_seed,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                    if payload is None:
                        raise RuntimeError(f"Exact-path worker did not publish {output}.")
    finally:
        lock.close()
    summary = audit.summarize_terminal(
        manifest_path=manifest_path,
        full_audit_path=full_audit_path,
    )
    return _publish_or_validate_audit(contract.terminal_audit_path(), summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen two-phase 707 continuous-rank calibration matrix."
    )
    parser.add_argument("phase", choices=("full-forward", "exact-path"))
    parser.add_argument("--manifest", type=Path, default=contract.MANIFEST_PATH)
    parser.add_argument(
        "--full-forward-audit",
        type=Path,
        default=contract.full_forward_audit_path(),
    )
    args = parser.parse_args()
    if args.phase == "full-forward":
        result = run_full_forward(args.manifest)
    else:
        result = run_exact_path(args.manifest, args.full_forward_audit)
    print(
        json.dumps(
            {
                "event": "continuous_rank_matrix_terminal",
                "phase": args.phase,
                "experiment_id": result["experiment_id"],
                "payload_sha256": result["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
