from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

FROZEN_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_run(output_dir: Path, scale: str, seed: int, steps: int) -> dict | None:
    summary_path = output_dir / f"{scale}-training.summary.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text())
    checkpoint = payload.get("checkpoint")
    if (
        payload.get("experiment_id") != "paper-grade-tier-s-training-v1"
        or payload.get("scale") != scale
        or payload.get("seed") != seed
        or payload.get("steps_completed") != steps
        or not isinstance(checkpoint, dict)
    ):
        return None
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.exists() or _sha256(checkpoint_path) != checkpoint["sha256"]:
        return None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe launcher for the preregistered P2 training matrix."
    )
    parser.add_argument("--scales", nargs="+", choices=("s55", "s151"), default=("s55", "s151"))
    parser.add_argument("--seeds", nargs="+", type=int, default=FROZEN_SEEDS)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training-matrix.json"),
    )
    args = parser.parse_args()
    if tuple(args.seeds) != FROZEN_SEEDS:
        raise ValueError(f"P2 seeds must exactly match the frozen list {FROZEN_SEEDS}.")
    if args.steps != 1000:
        raise ValueError("The frozen P2 training matrix requires exactly 1000 steps.")

    train_script = Path(__file__).with_name("train_m1_associative_recall.py")
    completed: list[dict] = []
    for scale in args.scales:
        for seed in args.seeds:
            output_dir = args.output_root / scale / f"seed-{seed}"
            payload = _completed_run(output_dir, scale, seed, args.steps)
            if payload is None:
                command = [
                    sys.executable,
                    str(train_script),
                    "--experiment-id",
                    "paper-grade-tier-s-training-v1",
                    "--scale",
                    scale,
                    "--seed",
                    str(seed),
                    "--steps",
                    str(args.steps),
                    "--minimum-steps",
                    str(args.steps),
                    "--output-dir",
                    str(output_dir),
                ]
                subprocess.run(command, check=True)
                payload = _completed_run(output_dir, scale, seed, args.steps)
                if payload is None:
                    raise RuntimeError(f"Training output failed validation: {scale}/{seed}.")
            completed.append(
                {
                    "scale": scale,
                    "seed": seed,
                    "initialization_seed": payload["initialization_seed"],
                    "data_order_seed": payload["data_order_seed"],
                    "training_evaluation_seed": payload["training_evaluation_seed"],
                    "steps_completed": payload["steps_completed"],
                    "checkpoint": payload["checkpoint"],
                    "summary": str(
                        args.output_root / scale / f"seed-{seed}" / f"{scale}-training.summary.json"
                    ),
                }
            )
            args.matrix_summary.parent.mkdir(parents=True, exist_ok=True)
            args.matrix_summary.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiment_id": "paper-grade-tier-s-training-matrix-v1",
                        "frozen_seeds": FROZEN_SEEDS,
                        "steps": args.steps,
                        "runs": completed,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
