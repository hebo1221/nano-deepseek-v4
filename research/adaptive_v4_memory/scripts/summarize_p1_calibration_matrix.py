from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

TRAINING_SEEDS = (6071401, 6071402, 6071403, 6071404, 6071405)
CALIBRATION_SEEDS = (7071401, 7071402, 7071403, 7071404, 7071405)
SCALES = ("s55", "s151")
EXPECTED_CONTEXTS = (80, 128, 256, 512, 1024)
EXPECTED_MULTIPLIERS = (1, 2, 4)
EXAMPLES_PER_FAMILY = 256
QUERY_TURNS_PER_CONVERSATION = {
    "single-remote-retrieval": 1,
    "multiple-independent-needles": 4,
    "associative-recall": 1,
    "multi-turn-query-shift": 4,
    "dense-global-aggregation": 8,
    "irrelevant-context-local-only": 1,
    "instruction-persistence": 4,
    "adversarial-lexical-distractors": 1,
    "long-generation-changing-evidence": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _pairs() -> set[tuple[str, int, int]]:
    return {
        (scale, training_seed, calibration_seed)
        for scale in SCALES
        for training_seed, calibration_seed in zip(
            TRAINING_SEEDS, CALIBRATION_SEEDS, strict=True
        )
    }


def _audit_slices(payload: dict[str, Any], path: Path) -> None:
    counts = payload.get("slice_example_counts", {})
    expected_keys = {
        f"{family}:{context}"
        for family in PAPER_GRADE_WORKLOAD_FAMILIES
        for context in EXPECTED_CONTEXTS
    }
    _require(set(counts) == expected_keys, f"Family/context slice drift: {path}")
    for family in PAPER_GRADE_WORKLOAD_FAMILIES:
        observed = sum(counts[f"{family}:{context}"] for context in EXPECTED_CONTEXTS)
        _require(observed == EXAMPLES_PER_FAMILY, f"Family sample count drift: {path}/{family}")
    for context in EXPECTED_CONTEXTS:
        context_counts = {
            counts[f"{family}:{context}"] for family in PAPER_GRADE_WORKLOAD_FAMILIES
        }
        _require(len(context_counts) == 1, f"Unbalanced family/context slices: {path}")


def _audit_raw(
    path: Path,
    *,
    matrix_run: dict[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    scale = matrix_run["scale"]
    training_seed = matrix_run["training_seed"]
    calibration_seed = matrix_run["calibration_seed"]
    _require(
        payload.get("experiment_id") == "p1-layer-quota-calibration-pilot-v1",
        f"Wrong raw experiment id: {path}",
    )
    _require(payload.get("scale") == scale, f"Scale drift: {path}")
    _require(payload.get("seed") == calibration_seed, f"Calibration seed drift: {path}")
    _require(payload.get("seed_namespace") == "calibration", f"Seed namespace drift: {path}")
    _require(
        tuple(payload.get("allowed_calibration_seeds", ())) == CALIBRATION_SEEDS,
        f"Allowed calibration seed drift: {path}",
    )
    _require(
        payload.get("source") == {"commit": source_commit, "dirty": False},
        f"Source provenance drift: {path}",
    )
    _require(
        payload.get("examples_per_family") == EXAMPLES_PER_FAMILY,
        f"Sample count drift: {path}",
    )
    _require(
        tuple(payload.get("families", ())) == PAPER_GRADE_WORKLOAD_FAMILIES,
        f"Workload family drift: {path}",
    )
    _require(
        tuple(payload.get("contexts", ())) == EXPECTED_CONTEXTS,
        f"Context grid drift: {path}",
    )
    leakage = payload.get("leakage_guard", {})
    _require(leakage.get("targets_used_for_quota_fit") is False, f"Target leakage: {path}")
    _require(
        leakage.get("held_out_evaluation_seed_used") is False,
        f"Evaluation-seed leakage: {path}",
    )
    _audit_slices(payload, path)

    layers = tuple(sorted(int(layer) for layer in payload["captured_queries_per_layer"]))
    expected_per_layer = EXAMPLES_PER_FAMILY * sum(QUERY_TURNS_PER_CONVERSATION.values())
    observed_by_layer = {
        int(layer): int(count)
        for layer, count in payload["captured_queries_per_layer"].items()
    }
    _require(
        all(observed_by_layer[layer] == expected_per_layer for layer in layers),
        f"Captured query count mismatch: {path}",
    )
    expected_total = expected_per_layer * len(layers)
    _require(payload.get("captured_query_count") == expected_total, f"Total query drift: {path}")
    _require(
        matrix_run.get("captured_query_count") == expected_total,
        f"Matrix/raw query count mismatch: {path}",
    )

    minimum = int(payload["minimum_blocks_per_layer"])
    calibrations = payload.get("calibrations", {})
    _require(set(calibrations) == {"1x", "2x", "4x"}, f"Budget grid drift: {path}")
    digests: set[str] = set()
    audited_calibrations: dict[str, Any] = {}
    for multiplier in EXPECTED_MULTIPLIERS:
        key = f"{multiplier}x"
        item = calibrations[key]
        signal = item["signal_config"]
        quota = item["quota"]
        normal = tuple((int(layer), int(budget)) for layer, budget in quota["layer_budgets"])
        dense = tuple(
            (int(layer), int(budget)) for layer, budget in quota["dense_layer_budgets"]
        )
        _require(tuple(layer for layer, _ in normal) == layers, f"Layer drift: {path}/{key}")
        _require(tuple(layer for layer, _ in dense) == layers, f"Dense layer drift: {path}/{key}")
        _require(
            tuple((int(layer), int(count)) for layer, count in quota["examples_per_layer"])
            == tuple((layer, expected_per_layer) for layer in layers),
            f"Quota sample count drift: {path}/{key}",
        )
        global_budget = minimum * len(layers) * multiplier
        _require(
            signal["global_block_budget"] == global_budget,
            f"Global budget drift: {path}/{key}",
        )
        _require(
            signal["max_extra_blocks_per_layer"] == minimum * (multiplier - 1),
            f"Per-layer allowance drift: {path}/{key}",
        )
        _require(
            all(budget >= minimum for _, budget in normal),
            f"Layer floor violated: {path}/{key}",
        )
        _require(
            sum(budget for _, budget in normal) <= global_budget,
            f"Global budget exceeded: {path}/{key}",
        )
        if multiplier == 1:
            _require(
                all(budget == minimum for _, budget in normal),
                f"The 1x fixed-policy floor is not uniform: {path}",
            )
        normal_by_layer = dict(normal)
        _require(
            all(budget >= normal_by_layer[layer] for layer, budget in dense),
            f"Dense quota below normal quota: {path}/{key}",
        )
        _require(
            sum(budget for _, budget in dense) <= signal["dense_fallback_block_budget"],
            f"Dense fallback budget exceeded: {path}/{key}",
        )
        digest = quota["calibration_digest"]
        _require(isinstance(digest, str) and len(digest) == 64, f"Invalid digest: {path}/{key}")
        _require(
            matrix_run["calibration_digests"].get(key) == digest,
            f"Matrix/raw calibration digest mismatch: {path}/{key}",
        )
        digests.add(digest)
        normal_values = tuple(budget for _, budget in normal)
        audited_calibrations[key] = {
            "layer_budgets": normal,
            "dense_layer_budgets": dense,
            "normal_total": sum(normal_values),
            "global_budget": global_budget,
            "global_budget_utilization": sum(normal_values) / global_budget,
            "non_uniform": len(set(normal_values)) > 1,
            "score_demand_quantiles": quota["score_demand_quantiles"],
            "candidate_demand_quantiles": quota["candidate_demand_quantiles"],
            "calibration_digest": digest,
        }
    _require(len(digests) == len(EXPECTED_MULTIPLIERS), f"Non-unique run digests: {path}")

    checkpoint = payload["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    _require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
    _require(checkpoint == matrix_run["checkpoint"], f"Matrix/raw checkpoint drift: {path}")
    _require(checkpoint_path.stat().st_size == checkpoint["bytes"], f"Checkpoint size drift: {path}")
    _require(_sha256(checkpoint_path) == checkpoint["sha256"], f"Checkpoint digest drift: {path}")
    _require(
        f"seed-{training_seed}" in checkpoint["path"],
        f"Training seed/checkpoint mismatch: {path}",
    )
    return {
        "scale": scale,
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "examples_per_family": EXAMPLES_PER_FAMILY,
        "conversations": EXAMPLES_PER_FAMILY * len(PAPER_GRADE_WORKLOAD_FAMILIES),
        "captured_queries_per_layer": expected_per_layer,
        "captured_query_count": expected_total,
        "layers": layers,
        "checkpoint": checkpoint,
        "raw_artifact": {"path": str(path), "sha256": _sha256(path)},
        "calibrations": audited_calibrations,
    }


def _aggregate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for budget, calibration in run["calibrations"].items():
            groups[(run["scale"], budget)].append(calibration)
    rows = []
    for (scale, budget), calibrations in sorted(groups.items()):
        totals = [item["normal_total"] for item in calibrations]
        utilizations = [item["global_budget_utilization"] for item in calibrations]
        vectors = {
            tuple(value for _, value in item["layer_budgets"]) for item in calibrations
        }
        rows.append(
            {
                "scale": scale,
                "budget": budget,
                "runs": len(calibrations),
                "unique_layer_budget_vectors": len(vectors),
                "non_uniform_runs": sum(bool(item["non_uniform"]) for item in calibrations),
                "normal_total_mean": statistics.fmean(totals),
                "normal_total_min": min(totals),
                "normal_total_max": max(totals),
                "global_budget_utilization_mean": statistics.fmean(utilizations),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the full P1 layer-quota matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/calibration-matrix.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = json.loads(args.matrix.read_text())
    _require(
        matrix.get("experiment_id") == "p1-layer-quota-calibration-matrix-v1",
        "Wrong matrix experiment id.",
    )
    _require(tuple(matrix.get("training_seeds", ())) == TRAINING_SEEDS, "Training seed drift.")
    _require(
        tuple(matrix.get("calibration_seeds", ())) == CALIBRATION_SEEDS,
        "Calibration seed drift.",
    )
    _require(matrix.get("examples_per_family") == EXAMPLES_PER_FAMILY, "Sample count drift.")
    source_commit = matrix.get("source_commit")
    _require(isinstance(source_commit, str) and len(source_commit) == 40, "Invalid source commit.")
    matrix_runs = matrix.get("runs", [])
    observed_pairs = {
        (run["scale"], run["training_seed"], run["calibration_seed"])
        for run in matrix_runs
    }
    _require(len(matrix_runs) == len(_pairs()), "Matrix does not contain exactly ten runs.")
    _require(observed_pairs == _pairs(), "Scale/training/calibration seed pairing drift.")

    audited = []
    for matrix_run in matrix_runs:
        raw = matrix_run["raw_artifact"]
        raw_path = Path(raw["path"])
        _require(raw_path.is_file(), f"Missing raw artifact: {raw_path}")
        _require(_sha256(raw_path) == raw["sha256"], f"Raw artifact digest drift: {raw_path}")
        audited.append(
            _audit_raw(raw_path, matrix_run=matrix_run, source_commit=source_commit)
        )
    audited.sort(key=lambda run: (run["scale"], run["training_seed"]))

    total_conversations = sum(run["conversations"] for run in audited)
    total_captured = sum(run["captured_query_count"] for run in audited)
    payload = {
        "schema_version": 1,
        "experiment_id": "p1-layer-quota-calibration-matrix-audit-v1",
        "interpretation": "five-seed, two-scale calibration-only result; no held-out quality claim",
        "source_commit": source_commit,
        "matrix_artifact": {"path": str(args.matrix), "sha256": _sha256(args.matrix)},
        "design": {
            "scales": SCALES,
            "training_seeds": TRAINING_SEEDS,
            "calibration_seeds": CALIBRATION_SEEDS,
            "families": PAPER_GRADE_WORKLOAD_FAMILIES,
            "contexts": EXPECTED_CONTEXTS,
            "examples_per_family_per_run": EXAMPLES_PER_FAMILY,
            "runs": len(audited),
            "total_conversations": total_conversations,
            "total_captured_query_observations": total_captured,
        },
        "validation": {
            "exact_seed_matrix_verified": True,
            "all_sources_clean_and_identical": True,
            "all_raw_digests_verified": True,
            "all_checkpoint_digests_verified": True,
            "all_family_context_slices_verified": True,
            "all_budget_bounds_verified": True,
            "uniform_1x_fixed_floor_verified": True,
            "targets_used_for_fit": False,
            "held_out_evaluation_seed_used": False,
        },
        "aggregate": _aggregate(audited),
        "runs": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"design": payload["design"], "validation": payload["validation"]}))


if __name__ == "__main__":
    main()
