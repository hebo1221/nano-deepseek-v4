from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_core_shard as shard
import torch

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

SCALES = ("s55", "s151")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _records_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _completed(
    output: Path,
    *,
    source_commit: str,
    scale: str,
    training_seed: int,
    family: str,
    context: int,
    replicate: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    calibration: Path,
    calibration_sha256: str,
    equivalence: Path,
    equivalence_sha256: str,
) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    payload = json.loads(output.read_text())
    if (
        payload.get("experiment_id") != "p2-core-quality-shard-v1"
        or payload.get("source") != {"commit": source_commit, "dirty": False}
        or payload.get("scale") != scale
        or payload.get("training_seed") != training_seed
        or payload.get("evaluation_seed") != shard._evaluation_seed(training_seed)
        or payload.get("family") != family
        or payload.get("context") != context
        or payload.get("replicate") != replicate
        or payload.get("examples") != shard.EXAMPLES_PER_SHARD
        or payload.get("chunk_size") != shard.CHUNK_SIZE
        or tuple(payload.get("policies", ())) != shard.CORE_POLICIES
    ):
        return None
    records = payload.get("records", [])
    if (
        not isinstance(records, list)
        or len(records) != shard.EXAMPLES_PER_SHARD * len(shard.CORE_POLICIES)
        or _records_digest(records) != payload.get("records_digest")
        or any(metric.get("budget_violations") != 0 for metric in payload["batch_metrics"])
    ):
        return None
    dependencies = (
        (payload["checkpoint"], checkpoint, checkpoint_sha256),
        (payload["calibration_artifact"], calibration, calibration_sha256),
        (payload["equivalence_artifact"], equivalence, equivalence_sha256),
    )
    if any(
        metadata.get("path") != str(path)
        or not path.is_file()
        or metadata.get("sha256") != expected_sha256
        for metadata, path, expected_sha256 in dependencies
    ):
        return None
    return payload


def _write_matrix(path: Path, source_commit: str, runs: list[dict[str, Any]]) -> None:
    runs.sort(
        key=lambda run: (
            run["scale"],
            run["training_seed"],
            run["family"],
            run["context"],
            run["replicate"],
        )
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "p2-core-quality-matrix-progress-v1",
        "source_commit": source_commit,
        "frozen_design": {
            "scales": SCALES,
            "training_seeds": shard.TRAINING_SEEDS,
            "evaluation_seeds": shard.EVALUATION_SEEDS,
            "families": PAPER_GRADE_WORKLOAD_FAMILIES,
            "contexts": shard.CONTEXTS,
            "replicates": shard.REPLICATES,
            "examples_per_shard": shard.EXAMPLES_PER_SHARD,
            "examples_per_family_checkpoint": (
                len(shard.CONTEXTS) * len(shard.REPLICATES) * shard.EXAMPLES_PER_SHARD
            ),
            "core_policies": shard.CORE_POLICIES,
            "chunk_size": shard.CHUNK_SIZE,
            "total_expected_shards": (
                len(SCALES)
                * len(shard.TRAINING_SEEDS)
                * len(PAPER_GRADE_WORKLOAD_FAMILIES)
                * len(shard.CONTEXTS)
                * len(shard.REPLICATES)
            ),
        },
        "completed_shards": len(runs),
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe frozen P2 core matrix runner.")
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument("--training-seed", type=int, action="append", choices=shard.TRAINING_SEEDS)
    parser.add_argument("--family", action="append", choices=PAPER_GRADE_WORKLOAD_FAMILIES)
    parser.add_argument("--context", type=int, action="append", choices=shard.CONTEXTS)
    parser.add_argument("--replicate", type=int, action="append", choices=shard.REPLICATES)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-shards", type=int)
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
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_core_quality"),
    )
    parser.add_argument(
        "--equivalence",
        type=Path,
        default=Path(
            "research/adaptive_v4_memory/results/"
            "p1-chunked-cache-equivalence.summary.json"
        ),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P2 core matrix requires CUDA.")
    if args.max_new_shards is not None and args.max_new_shards <= 0:
        raise ValueError("max-new-shards must be positive.")
    if args.batch_size <= 0 or shard.EXAMPLES_PER_SHARD % args.batch_size != 0:
        raise ValueError("batch-size must divide 20.")
    if _dirty():
        raise RuntimeError("P2 core matrix requires a clean source tree.")
    source_commit = _head()
    equivalence = shard._equivalence(args.equivalence)
    equivalence_sha256 = _sha256(args.equivalence)
    scales = tuple(args.scale or SCALES)
    training_seeds = tuple(args.training_seed or shard.TRAINING_SEEDS)
    families = tuple(args.family or PAPER_GRADE_WORKLOAD_FAMILIES)
    contexts = tuple(args.context or shard.CONTEXTS)
    replicates = tuple(args.replicate or shard.REPLICATES)
    completed_runs: list[dict[str, Any]] = []
    new_shards = 0

    for scale in scales:
        for training_seed in training_seeds:
            checkpoint = (
                args.training_root
                / scale
                / f"seed-{training_seed}"
                / f"{scale}-step-1000.pt"
            )
            calibration_path = (
                args.calibration_root
                / scale
                / f"seed-{training_seed}"
                / "p1-layer-quotas.json"
            )
            calibration = heldout._load_calibration(
                calibration_path, checkpoint, scale
            )
            checkpoint_sha256 = calibration["checkpoint"]["sha256"]
            calibration_sha256 = _sha256(calibration_path)
            model: Any = None
            for family in families:
                for context in contexts:
                    for replicate in replicates:
                        output = (
                            args.output_root
                            / scale
                            / f"seed-{training_seed}"
                            / family
                            / f"context-{context}"
                            / f"replicate-{replicate}.json"
                        )
                        payload = _completed(
                            output,
                            source_commit=source_commit,
                            scale=scale,
                            training_seed=training_seed,
                            family=family,
                            context=context,
                            replicate=replicate,
                            checkpoint=checkpoint,
                            checkpoint_sha256=checkpoint_sha256,
                            calibration=calibration_path,
                            calibration_sha256=calibration_sha256,
                            equivalence=args.equivalence,
                            equivalence_sha256=equivalence_sha256,
                        )
                        if payload is None:
                            if args.max_new_shards is not None and new_shards >= args.max_new_shards:
                                _write_matrix(args.matrix_summary, source_commit, completed_runs)
                                return
                            if model is None:
                                model = pilot._load_model(checkpoint)
                            payload = shard.build_payload(
                                model,
                                checkpoint=checkpoint,
                                calibration_path=calibration_path,
                                calibration=calibration,
                                equivalence_path=args.equivalence,
                                equivalence=equivalence,
                                scale=scale,
                                family=family,
                                context=context,
                                replicate=replicate,
                                training_seed=training_seed,
                                batch_size=args.batch_size,
                                checkpoint_sha256=checkpoint_sha256,
                                calibration_sha256=calibration_sha256,
                                equivalence_sha256=equivalence_sha256,
                                source_state={"commit": source_commit, "dirty": False},
                            )
                            output.parent.mkdir(parents=True, exist_ok=True)
                            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                            new_shards += 1
                            print(
                                json.dumps(
                                    {
                                        "completed": str(output),
                                        "new_shards": new_shards,
                                        "wall_seconds": payload["wall_seconds"],
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                        completed_runs.append(
                            {
                                "scale": scale,
                                "training_seed": training_seed,
                                "evaluation_seed": payload["evaluation_seed"],
                                "family": family,
                                "context": context,
                                "replicate": replicate,
                                "raw_artifact": {
                                    "path": str(output),
                                    "sha256": _sha256(output),
                                },
                                "records_digest": payload["records_digest"],
                                "wall_seconds": payload["wall_seconds"],
                            }
                        )
                        _write_matrix(args.matrix_summary, source_commit, completed_runs)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
