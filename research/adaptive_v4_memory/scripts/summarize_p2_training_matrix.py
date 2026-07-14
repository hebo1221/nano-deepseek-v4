from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

FROZEN_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
FROZEN_SCALES = ("s55", "s151")
FROZEN_STEPS = 1000
EXPERIMENT_ID = "paper-grade-tier-s-training-matrix-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _distribution(values: list[float]) -> dict[str, float]:
    _require(bool(values), "Cannot summarize an empty distribution.")
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def audit_matrix(matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text())
    _require(matrix.get("experiment_id") == EXPERIMENT_ID, "Wrong experiment id.")
    _require(tuple(matrix.get("frozen_seeds", ())) == FROZEN_SEEDS, "Seed drift.")
    _require(matrix.get("steps") == FROZEN_STEPS, "Training-step drift.")
    runs = matrix.get("runs")
    _require(isinstance(runs, list), "Matrix runs must be a list.")

    expected = {(scale, seed) for scale in FROZEN_SCALES for seed in FROZEN_SEEDS}
    observed = {(run.get("scale"), run.get("seed")) for run in runs}
    _require(len(runs) == len(expected), "The matrix must contain exactly ten runs.")
    _require(observed == expected, "The matrix has a missing, duplicate, or extra run.")

    audited_runs: list[dict[str, Any]] = []
    source_commits: set[str] = set()
    for scale in FROZEN_SCALES:
        for seed in FROZEN_SEEDS:
            run = next(
                item for item in runs if item.get("scale") == scale and item.get("seed") == seed
            )
            summary_path = Path(run["summary"])
            _require(summary_path.is_file(), f"Missing summary: {summary_path}")
            summary = json.loads(summary_path.read_text())
            _require(summary.get("scale") == scale, f"Scale mismatch: {summary_path}")
            _require(summary.get("seed") == seed, f"Seed mismatch: {summary_path}")
            _require(
                summary.get("experiment_id") == "paper-grade-tier-s-training-v1",
                f"Wrong run experiment id: {summary_path}",
            )
            _require(
                run.get("steps_completed") == FROZEN_STEPS
                and summary.get("steps_completed") == FROZEN_STEPS,
                f"Incomplete run: {scale}/{seed}",
            )
            _require(summary.get("initialization_seed") == seed, "Initialization seed drift.")
            _require(summary.get("data_order_seed") == seed + 1, "Data-order seed drift.")
            _require(
                summary.get("training_evaluation_seed") == seed + 10_000,
                "Training-evaluation seed drift.",
            )
            source = summary.get("source", {})
            _require(source.get("dirty") is False, f"Dirty source run: {scale}/{seed}")
            source_commit = source.get("commit")
            if not isinstance(source_commit, str) or not source_commit:
                raise ValueError("Missing source commit.")
            source_commits.add(source_commit)

            history = summary.get("history")
            if not isinstance(history, list) or not history:
                raise ValueError(f"Missing history: {scale}/{seed}")
            final = history[-1]
            _require(final.get("step") == FROZEN_STEPS, f"Wrong final step: {scale}/{seed}")

            checkpoint = summary.get("checkpoint", {})
            _require(
                checkpoint == run.get("checkpoint"), f"Checkpoint metadata drift: {scale}/{seed}"
            )
            checkpoint_path = Path(checkpoint["path"])
            _require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
            _require(
                checkpoint_path.stat().st_size == checkpoint["bytes"],
                f"Checkpoint byte mismatch: {checkpoint_path}",
            )
            actual_checkpoint_sha = _sha256(checkpoint_path)
            _require(
                actual_checkpoint_sha == checkpoint["sha256"],
                f"Checkpoint digest mismatch: {checkpoint_path}",
            )

            native = final["native"]
            audited_runs.append(
                {
                    "scale": scale,
                    "seed": seed,
                    "initialization_seed": summary["initialization_seed"],
                    "data_order_seed": summary["data_order_seed"],
                    "training_evaluation_seed": summary["training_evaluation_seed"],
                    "reported_stopped_early": summary.get("stopped_early"),
                    "final_native_accuracy": native["mean_accuracy"],
                    "final_native_accuracy_by_length": {
                        "48": native["accuracy_length_48"],
                        "64": native["accuracy_length_64"],
                        "80": native["accuracy_length_80"],
                    },
                    "training_elapsed_seconds": summary["runtime"]["elapsed_seconds"],
                    "peak_allocated_bytes": summary["runtime"]["peak_allocated_bytes"],
                    "peak_reserved_bytes": summary["runtime"]["peak_reserved_bytes"],
                    "summary": {
                        "path": str(summary_path),
                        "sha256": _sha256(summary_path),
                    },
                    "checkpoint": checkpoint,
                }
            )

    _require(len(source_commits) == 1, "Runs were produced from multiple source commits.")
    scale_summaries: dict[str, Any] = {}
    for scale in FROZEN_SCALES:
        scale_runs = [run for run in audited_runs if run["scale"] == scale]
        scale_summaries[scale] = {
            "final_native_accuracy": _distribution(
                [run["final_native_accuracy"] for run in scale_runs]
            ),
            "final_native_accuracy_by_length": {
                length: _distribution(
                    [run["final_native_accuracy_by_length"][length] for run in scale_runs]
                )
                for length in ("48", "64", "80")
            },
            "training_elapsed_seconds": _distribution(
                [run["training_elapsed_seconds"] for run in scale_runs]
            ),
        }

    return {
        "schema_version": 1,
        "experiment_id": "paper-grade-tier-s-training-audit-v1",
        "interpretation": (
            "Training-matrix audit only; controller quality conclusions require the frozen P2 "
            "held-out workload evaluation."
        ),
        "matrix": {
            "path": str(matrix_path),
            "sha256": _sha256(matrix_path),
        },
        "source_commit": next(iter(source_commits)),
        "validation": {
            "expected_runs": len(expected),
            "observed_runs": len(audited_runs),
            "all_runs_complete": True,
            "all_sources_clean": True,
            "all_checkpoint_digests_verified": True,
            "low_quality_runs_excluded": 0,
        },
        "scales": scale_summaries,
        "runs": audited_runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen P2 training matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training-matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_matrix(args.matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["validation"], sort_keys=True))


if __name__ == "__main__":
    main()
