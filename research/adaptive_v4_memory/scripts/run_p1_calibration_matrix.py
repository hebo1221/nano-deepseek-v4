from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
CALIBRATION_SEEDS = (7071401, 7071402, 7071403, 7071404, 7071405)
SCALES = ("s55", "s151")
EXAMPLES_PER_FAMILY = 256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _completed(
    output: Path,
    *,
    scale: str,
    calibration_seed: int,
    checkpoint: Path,
    source_commit: str,
) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    payload = json.loads(output.read_text())
    if (
        payload.get("experiment_id") != "p1-layer-quota-calibration-pilot-v1"
        or payload.get("scale") != scale
        or payload.get("seed") != calibration_seed
        or payload.get("examples_per_family") != EXAMPLES_PER_FAMILY
        or payload.get("source") != {"commit": source_commit, "dirty": False}
    ):
        return None
    checkpoint_metadata = payload.get("checkpoint", {})
    if (
        checkpoint_metadata.get("path") != str(checkpoint)
        or not checkpoint.is_file()
        or checkpoint_metadata.get("bytes") != checkpoint.stat().st_size
        or checkpoint_metadata.get("sha256") != _sha256(checkpoint)
    ):
        return None
    if set(payload.get("calibrations", {})) != {"1x", "2x", "4x"}:
        return None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe five-seed, two-scale P1 quota calibration matrix."
    )
    parser.add_argument("--examples-per-family", type=int, default=EXAMPLES_PER_FAMILY)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/calibration_matrix"),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/calibration-matrix.json"),
    )
    args = parser.parse_args()
    if args.examples_per_family != EXAMPLES_PER_FAMILY:
        raise ValueError(
            f"The frozen calibration matrix requires {EXAMPLES_PER_FAMILY} examples per family."
        )
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    source_commit = _head()
    calibration_script = Path(__file__).with_name("calibrate_p1_layer_quotas.py")
    completed: list[dict[str, Any]] = []
    for scale in SCALES:
        for training_seed, calibration_seed in zip(TRAINING_SEEDS, CALIBRATION_SEEDS, strict=True):
            checkpoint = (
                args.training_root / scale / f"seed-{training_seed}" / f"{scale}-step-1000.pt"
            )
            output = args.output_root / scale / f"seed-{training_seed}" / "p1-layer-quotas.json"
            payload = _completed(
                output,
                scale=scale,
                calibration_seed=calibration_seed,
                checkpoint=checkpoint,
                source_commit=source_commit,
            )
            if payload is None:
                command = [
                    sys.executable,
                    str(calibration_script),
                    "--checkpoint",
                    str(checkpoint),
                    "--scale",
                    scale,
                    "--output",
                    str(output),
                    "--examples-per-family",
                    str(EXAMPLES_PER_FAMILY),
                    "--batch-size",
                    str(args.batch_size),
                    "--seed",
                    str(calibration_seed),
                ]
                subprocess.run(command, check=True)
                payload = _completed(
                    output,
                    scale=scale,
                    calibration_seed=calibration_seed,
                    checkpoint=checkpoint,
                    source_commit=source_commit,
                )
                if payload is None:
                    raise RuntimeError(
                        f"Calibration output failed validation: {scale}/{training_seed}."
                    )
            completed.append(
                {
                    "scale": scale,
                    "training_seed": training_seed,
                    "calibration_seed": calibration_seed,
                    "checkpoint": payload["checkpoint"],
                    "raw_artifact": {
                        "path": str(output),
                        "sha256": _sha256(output),
                    },
                    "captured_query_count": payload["captured_query_count"],
                    "calibration_digests": {
                        key: value["quota"]["calibration_digest"]
                        for key, value in payload["calibrations"].items()
                    },
                }
            )
            args.matrix_summary.parent.mkdir(parents=True, exist_ok=True)
            args.matrix_summary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "p1-layer-quota-calibration-matrix-v1",
                        "source_commit": source_commit,
                        "training_seeds": TRAINING_SEEDS,
                        "calibration_seeds": CALIBRATION_SEEDS,
                        "examples_per_family": EXAMPLES_PER_FAMILY,
                        "batch_size": args.batch_size,
                        "runs": completed,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
