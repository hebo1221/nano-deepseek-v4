from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from itertools import product
from pathlib import Path
from typing import Any

import benchmark_m5_online_controller as pilot
import evaluate_p1_heldout_policy_pilot as heldout
import evaluate_p2_causal_factorial_shard as shard
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

SCALES = ("s55", "s151")
EXPECTED_P2_SHARDS = 4_500
EXPECTED_CAUSAL_SHARDS = (
    len(SCALES)
    * len(shard.TRAINING_SEEDS)
    * len(shard.BUDGET_LABELS)
    * len(PAPER_GRADE_WORKLOAD_FAMILIES)
    * len(shard.CONTEXTS)
    * len(shard.REPLICATES)
)


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


def require_p2_audit(matrix_path: Path, audit_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text())
    design = matrix.get("frozen_design", {})
    runs = matrix.get("runs", [])
    expected = set(
        product(
            SCALES,
            shard.TRAINING_SEEDS,
            PAPER_GRADE_WORKLOAD_FAMILIES,
            shard.CONTEXTS,
            shard.REPLICATES,
        )
    )
    seen = {
        (
            run.get("scale"),
            run.get("training_seed"),
            run.get("family"),
            run.get("context"),
            run.get("replicate"),
        )
        for run in runs
    }
    if (
        matrix.get("experiment_id") != "p2-core-quality-matrix-progress-v1"
        or matrix.get("completed_shards") != EXPECTED_P2_SHARDS
        or design.get("total_expected_shards") != EXPECTED_P2_SHARDS
        or len(runs) != EXPECTED_P2_SHARDS
        or seen != expected
    ):
        raise RuntimeError(
            "The causal factorial is deferred until the frozen P2 core has "
            f"4500 unique shards ({matrix.get('completed_shards', 0)}/4500 recorded)."
        )
    audit = json.loads(audit_path.read_text())
    raw = audit.get("raw_matrix", {})
    checks = audit.get("audit", {})
    if (
        audit.get("experiment_id") != "p2-core-quality-matrix-audit-v1"
        or raw.get("path") != str(matrix_path)
        or raw.get("sha256") != _sha256(matrix_path)
        or checks.get("all_raw_shards_verified") is not True
        or checks.get("all_dependency_digests_verified") is not True
        or checks.get("all_record_digests_verified") is not True
        or checks.get("no_budget_violations") is not True
        or checks.get("unique_shards") != EXPECTED_P2_SHARDS
    ):
        raise RuntimeError("The full digest-bound P2 core audit is required before causality.")
    return audit


def _records_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _completed(
    output: Path,
    *,
    implementation_digest: str,
    scale: str,
    budget: str,
    training_seed: int,
    family: str,
    context: int,
    replicate: int,
    checkpoint: Path,
    checkpoint_sha256: str,
    calibration: Path,
    calibration_sha256: str,
    memory_match: Path,
    memory_match_sha256: str,
    equivalence: Path,
    equivalence_sha256: str,
    design: Path,
    design_sha256: str,
) -> dict[str, Any] | None:
    if not output.is_file():
        return None
    payload = json.loads(output.read_text())
    if (
        payload.get("experiment_id") != "p2-causal-factorial-shard-v1"
        or payload.get("source", {}).get("dirty") is not False
        or payload.get("source", {}).get("implementation_digest") != implementation_digest
        or payload.get("scale") != scale
        or payload.get("budget") != budget
        or payload.get("training_seed") != training_seed
        or payload.get("evaluation_seed") != shard.core._evaluation_seed(training_seed)
        or payload.get("family") != family
        or payload.get("context") != context
        or payload.get("replicate") != replicate
        or payload.get("examples") != shard.EXAMPLES_PER_SHARD
        or payload.get("batch_size") != shard.BATCH_SIZE
        or payload.get("chunk_size") != shard.CHUNK_SIZE_BY_SCALE[scale]
        or tuple(payload.get("primary_arms", ())) != shard.PRIMARY_ARM_NAMES
        or tuple(payload.get("supplemental_baseline_arms", ()))
        != shard.SUPPLEMENTAL_BASELINE_ARM_NAMES
        or tuple(payload.get("component_arms", ())) != shard.COMPONENT_ARM_NAMES
        or tuple(payload.get("physical_arms", ())) != shard.PHYSICAL_ARM_NAMES
    ):
        return None
    records = payload.get("records", [])
    metrics = payload.get("batch_metrics", [])
    physical = payload.get("physical_measurements", [])
    if (
        not isinstance(records, list)
        or len(records) != shard.EXAMPLES_PER_SHARD * len(shard.ALL_ARM_NAMES)
        or _records_digest(records) != payload.get("records_digest")
        or len(metrics) != shard.BATCHES_PER_SHARD * len(shard.ALL_ARM_NAMES)
        or any(metric.get("budget_violations") != 0 for metric in metrics)
        or len(physical) != shard.BATCHES_PER_SHARD * len(shard.PHYSICAL_ARM_NAMES)
        or any(item.get("predictions_identical_to_chunked") is not True for item in physical)
    ):
        return None
    dependencies = (
        (payload.get("checkpoint", {}), checkpoint, checkpoint_sha256),
        (payload.get("calibration_artifact", {}), calibration, calibration_sha256),
        (payload.get("memory_match_artifact", {}), memory_match, memory_match_sha256),
        (payload.get("equivalence_artifact", {}), equivalence, equivalence_sha256),
        (payload.get("design_manifest", {}), design, design_sha256),
    )
    if any(
        metadata.get("path") != str(path)
        or not path.is_file()
        or metadata.get("sha256") != expected_sha256
        for metadata, path, expected_sha256 in dependencies
    ):
        return None
    return payload


def _write_matrix(
    path: Path,
    *,
    source_commit: str,
    implementation_digest: str,
    p2_matrix: Path,
    p2_audit: Path,
    design: Path,
    runs: list[dict[str, Any]],
) -> None:
    runs.sort(
        key=lambda run: (
            run["scale"],
            run["training_seed"],
            run["budget"],
            run["family"],
            run["context"],
            run["replicate"],
        )
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "p2-causal-factorial-matrix-progress-v1",
        "source_commit": source_commit,
        "implementation_digest": implementation_digest,
        "prerequisites": {
            "p2_matrix": {"path": str(p2_matrix), "sha256": _sha256(p2_matrix)},
            "p2_audit": {"path": str(p2_audit), "sha256": _sha256(p2_audit)},
            "design": {"path": str(design), "sha256": _sha256(design)},
        },
        "frozen_design": {
            "scales": SCALES,
            "training_seeds": shard.TRAINING_SEEDS,
            "evaluation_seeds": shard.EVALUATION_SEEDS,
            "budgets": shard.BUDGET_LABELS,
            "families": PAPER_GRADE_WORKLOAD_FAMILIES,
            "contexts": shard.CONTEXTS,
            "replicates": shard.REPLICATES,
            "examples_per_shard": shard.EXAMPLES_PER_SHARD,
            "batch_size": shard.BATCH_SIZE,
            "examples_per_family_checkpoint_budget": (
                len(shard.CONTEXTS) * len(shard.REPLICATES) * shard.EXAMPLES_PER_SHARD
            ),
            "primary_arms": shard.PRIMARY_ARM_NAMES,
            "supplemental_baseline_arms": shard.SUPPLEMENTAL_BASELINE_ARM_NAMES,
            "component_arms": shard.COMPONENT_ARM_NAMES,
            "physical_arms": shard.PHYSICAL_ARM_NAMES,
            "chunk_size_by_scale": shard.CHUNK_SIZE_BY_SCALE,
            "total_expected_shards": EXPECTED_CAUSAL_SHARDS,
            "total_quality_policy_conversations": (
                EXPECTED_CAUSAL_SHARDS * shard.EXAMPLES_PER_SHARD * len(shard.ALL_ARM_NAMES)
            ),
            "total_physical_policy_conversations": (
                EXPECTED_CAUSAL_SHARDS * shard.EXAMPLES_PER_SHARD * len(shard.PHYSICAL_ARM_NAMES)
            ),
        },
        "completed_shards": len(runs),
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume-safe frozen P2 causal-factorial matrix runner."
    )
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument("--training-seed", type=int, action="append", choices=shard.TRAINING_SEEDS)
    parser.add_argument("--budget", action="append", choices=shard.BUDGET_LABELS)
    parser.add_argument("--family", action="append", choices=PAPER_GRADE_WORKLOAD_FAMILIES)
    parser.add_argument("--context", type=int, action="append", choices=shard.CONTEXTS)
    parser.add_argument("--replicate", type=int, action="append", choices=shard.REPLICATES)
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
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2_causal_factorial"),
    )
    parser.add_argument(
        "--memory-match-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2_causal_hot_memory"
        ),
    )
    parser.add_argument(
        "--equivalence-root",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2_causal_equivalence"
        ),
    )
    parser.add_argument(
        "--design",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json"),
    )
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--p2-audit",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.summary.json"
        ),
    )
    parser.add_argument(
        "--matrix-summary",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-factorial-matrix.json"),
    )
    args = parser.parse_args()
    if args.max_new_shards is not None and args.max_new_shards <= 0:
        raise ValueError("max-new-shards must be positive.")
    if _dirty():
        raise RuntimeError("P2 causal-factorial matrix requires a clean source tree.")
    require_p2_audit(args.p2_matrix, args.p2_audit)
    design = json.loads(args.design.read_text())
    if (
        design.get("experiment_id") != "p2-causal-factorial-v1"
        or design.get("execution", {}).get("total_expected_shards") != EXPECTED_CAUSAL_SHARDS
    ):
        raise RuntimeError("The causal-factorial design does not freeze 9000 shards.")
    gpu_lock = acquire_gpu_lock("p2-causal-factorial-matrix")
    if not torch.cuda.is_available():
        raise RuntimeError("P2 causal-factorial matrix requires CUDA.")
    source_commit = _head()
    implementation_digest = shard.implementation_digest()
    scales = tuple(args.scale or SCALES)
    training_seeds = tuple(args.training_seed or shard.TRAINING_SEEDS)
    budgets = tuple(args.budget or shard.BUDGET_LABELS)
    families = tuple(args.family or PAPER_GRADE_WORKLOAD_FAMILIES)
    contexts = tuple(args.context or shard.CONTEXTS)
    replicates = tuple(args.replicate or shard.REPLICATES)
    design_sha256 = _sha256(args.design)
    completed_runs: list[dict[str, Any]] = []
    new_shards = 0
    for scale in scales:
        for training_seed in training_seeds:
            checkpoint = (
                args.training_root / scale / f"seed-{training_seed}" / f"{scale}-step-1000.pt"
            )
            calibration_path = (
                args.calibration_root / scale / f"seed-{training_seed}" / "p1-layer-quotas.json"
            )
            calibration = heldout._load_calibration(calibration_path, checkpoint, scale)
            checkpoint_sha256 = calibration["checkpoint"]["sha256"]
            calibration_sha256 = _sha256(calibration_path)
            memory_match_path = (
                args.memory_match_root
                / scale
                / f"seed-{training_seed}"
                / "p2-causal-hot-memory-match.summary.json"
            )
            memory_match = shard._memory_match(
                memory_match_path,
                scale=scale,
                training_seed=training_seed,
                calibration_path=calibration_path,
            )
            memory_match_sha256 = _sha256(memory_match_path)
            equivalence_path = (
                args.equivalence_root / scale / f"seed-{training_seed}.summary.json"
            )
            equivalence = shard._equivalence(
                equivalence_path, scale, training_seed=training_seed
            )
            equivalence_sha256 = _sha256(equivalence_path)
            model: Any = None
            for budget in budgets:
                for family in families:
                    for context in contexts:
                        for replicate in replicates:
                            output = (
                                args.output_root
                                / scale
                                / f"seed-{training_seed}"
                                / f"budget-{budget}"
                                / family
                                / f"context-{context}"
                                / f"replicate-{replicate}.json"
                            )
                            payload = _completed(
                                output,
                                implementation_digest=implementation_digest,
                                scale=scale,
                                budget=budget,
                                training_seed=training_seed,
                                family=family,
                                context=context,
                                replicate=replicate,
                                checkpoint=checkpoint,
                                checkpoint_sha256=checkpoint_sha256,
                                calibration=calibration_path,
                                calibration_sha256=calibration_sha256,
                                memory_match=memory_match_path,
                                memory_match_sha256=memory_match_sha256,
                                equivalence=equivalence_path,
                                equivalence_sha256=equivalence_sha256,
                                design=args.design,
                                design_sha256=design_sha256,
                            )
                            if payload is None:
                                if (
                                    args.max_new_shards is not None
                                    and new_shards >= args.max_new_shards
                                ):
                                    _write_matrix(
                                        args.matrix_summary,
                                        source_commit=source_commit,
                                        implementation_digest=implementation_digest,
                                        p2_matrix=args.p2_matrix,
                                        p2_audit=args.p2_audit,
                                        design=args.design,
                                        runs=completed_runs,
                                    )
                                    gpu_lock.close()
                                    return
                                if model is None:
                                    model = pilot._load_model(checkpoint)
                                payload = shard.build_payload(
                                    model,
                                    checkpoint=checkpoint,
                                    calibration_path=calibration_path,
                                    calibration=calibration,
                                    memory_match_path=memory_match_path,
                                    memory_match=memory_match,
                                    equivalence_path=equivalence_path,
                                    equivalence=equivalence,
                                    design_path=args.design,
                                    scale=scale,
                                    budget_label=budget,
                                    family=family,
                                    context=context,
                                    replicate=replicate,
                                    training_seed=training_seed,
                                    batch_size=shard.BATCH_SIZE,
                                    checkpoint_sha256=checkpoint_sha256,
                                    calibration_sha256=calibration_sha256,
                                    memory_match_sha256=memory_match_sha256,
                                    equivalence_sha256=equivalence_sha256,
                                    design_sha256=design_sha256,
                                    source={
                                        "commit": source_commit,
                                        "dirty": False,
                                        "implementation_digest": implementation_digest,
                                    },
                                )
                                output.parent.mkdir(parents=True, exist_ok=True)
                                output.write_text(
                                    json.dumps(payload, indent=2, sort_keys=True) + "\n"
                                )
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
                                    "budget": budget,
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
                            _write_matrix(
                                args.matrix_summary,
                                source_commit=source_commit,
                                implementation_digest=implementation_digest,
                                p2_matrix=args.p2_matrix,
                                p2_audit=args.p2_audit,
                                design=args.design,
                                runs=completed_runs,
                            )
            del model
            torch.cuda.empty_cache()
    gpu_lock.close()


if __name__ == "__main__":
    main()
