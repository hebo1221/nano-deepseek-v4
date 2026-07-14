from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import evaluate_p2_causal_factorial_shard as shard
import run_p2_causal_factorial_matrix as matrix

SCALES = ("s55", "s151")
SCRIPT_ROOT = Path(__file__).resolve().parent


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _valid_memory_match(
    path: Path,
    *,
    scale: str,
    training_seed: int,
    calibration: Path,
) -> bool:
    try:
        shard._memory_match(
            path,
            scale=scale,
            training_seed=training_seed,
            calibration_path=calibration,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def _valid_equivalence(path: Path, *, scale: str, training_seed: int) -> bool:
    try:
        shard._equivalence(path, scale, training_seed=training_seed)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all post-P2 physical-match and exact-equivalence prerequisites."
    )
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument(
        "--training-seed", type=int, action="append", choices=shard.TRAINING_SEEDS
    )
    parser.add_argument("--max-new-stages", type=int)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/calibration_matrix"),
    )
    parser.add_argument(
        "--memory-match-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_hot_memory"),
    )
    parser.add_argument(
        "--equivalence-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_equivalence"),
    )
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--p2-audit",
        type=Path,
        default=Path("research/adaptive_v4_memory/results/p2-core-quality-matrix.summary.json"),
    )
    args = parser.parse_args()
    if args.max_new_stages is not None and args.max_new_stages <= 0:
        raise ValueError("max-new-stages must be positive.")
    if matrix._dirty():
        raise RuntimeError("Causal prerequisite orchestration requires a clean source tree.")
    matrix.require_p2_audit(args.p2_matrix, args.p2_audit)
    scales = tuple(args.scale or SCALES)
    seeds = tuple(args.training_seed or shard.TRAINING_SEEDS)
    new_stages = 0
    for scale in scales:
        for training_seed in seeds:
            checkpoint = (
                args.training_root / scale / f"seed-{training_seed}" / f"{scale}-step-1000.pt"
            )
            calibration = (
                args.calibration_root / scale / f"seed-{training_seed}" / "p1-layer-quotas.json"
            )
            match_dir = args.memory_match_root / scale / f"seed-{training_seed}"
            match_raw = match_dir / "p2-causal-hot-memory-match.raw.json"
            match_summary = match_dir / "p2-causal-hot-memory-match.summary.json"
            if not _valid_memory_match(
                match_summary,
                scale=scale,
                training_seed=training_seed,
                calibration=calibration,
            ):
                if args.max_new_stages is not None and new_stages >= args.max_new_stages:
                    return
                _run(
                    [
                        sys.executable,
                        str(SCRIPT_ROOT / "calibrate_p2_causal_hot_memory.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--calibration",
                        str(calibration),
                        "--scale",
                        scale,
                        "--raw-output",
                        str(match_raw),
                        "--summary-output",
                        str(match_summary),
                        "--p2-matrix",
                        str(args.p2_matrix),
                        "--p2-audit",
                        str(args.p2_audit),
                    ]
                )
                new_stages += 1
            equivalence_dir = args.equivalence_root / scale
            equivalence_raw = equivalence_dir / f"seed-{training_seed}.raw.json"
            equivalence_summary = equivalence_dir / f"seed-{training_seed}.summary.json"
            if not _valid_equivalence(
                equivalence_summary, scale=scale, training_seed=training_seed
            ):
                if args.max_new_stages is not None and new_stages >= args.max_new_stages:
                    return
                _run(
                    [
                        sys.executable,
                        str(SCRIPT_ROOT / "validate_p2_causal_factorial_equivalence.py"),
                        "--checkpoint",
                        str(checkpoint),
                        "--calibration",
                        str(calibration),
                        "--memory-match",
                        str(match_summary),
                        "--scale",
                        scale,
                        "--raw-output",
                        str(equivalence_raw),
                        "--summary-output",
                        str(equivalence_summary),
                        "--p2-matrix",
                        str(args.p2_matrix),
                        "--p2-audit",
                        str(args.p2_audit),
                    ]
                )
                new_stages += 1
            print(
                json.dumps(
                    {
                        "scale": scale,
                        "training_seed": training_seed,
                        "memory_match": str(match_summary),
                        "equivalence": str(equivalence_summary),
                        "new_stages": new_stages,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
