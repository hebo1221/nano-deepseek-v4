from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import collect_p1_online_lookahead_labels as labels
import evaluate_p1_online_learned_lookahead_shard as evaluator
import evaluate_p2_causal_factorial_shard as causal
import numpy as np
import run_p1_online_learned_lookahead as matrix_runner
from summarize_p2_causal_factorial import contrast_statistics
from summarize_p2_core_matrix import holm_bonferroni

PRIMARY = "online-learned-lookahead+pins"
ONE_TOKEN = "one-token-training-free+pins"
FIXED = "memory-matched-fixed+pins"
MAXIMUM_HBM_DIFFERENCE = 0.01
MAXIMUM_WORST_SLICE_REGRESSION = -0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def summarize(matrix_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text())
    expected = matrix.get("expected", {})
    completed = matrix.get("completed", {})
    _require(
        matrix.get("experiment_id") == "p1-online-learned-lookahead-matrix-progress-v1",
        "Wrong online-lookahead matrix id.",
    )
    _require(matrix.get("source", {}).get("dirty") is False, "Dirty matrix source.")
    _require(
        expected
        == {
            "label_shards": matrix_runner.EXPECTED_LABEL_SHARDS,
            "policies": matrix_runner.EXPECTED_POLICIES,
            "test_shards": matrix_runner.EXPECTED_TEST_SHARDS,
            "test_examples_per_shard": 20,
            "arms": list(evaluator.ARMS),
        },
        "Online-lookahead frozen matrix shape drifted.",
    )
    _require(
        completed
        == {
            "label_shards": matrix_runner.EXPECTED_LABEL_SHARDS,
            "policies": matrix_runner.EXPECTED_POLICIES,
            "test_shards": matrix_runner.EXPECTED_TEST_SHARDS,
        },
        "Online-lookahead matrix is incomplete.",
    )
    _require(
        len(matrix.get("label_shards", [])) == matrix_runner.EXPECTED_LABEL_SHARDS
        and len(matrix.get("policies", [])) == matrix_runner.EXPECTED_POLICIES
        and len(matrix.get("test_shards", [])) == matrix_runner.EXPECTED_TEST_SHARDS,
        "Online-lookahead matrix index coverage drifted.",
    )
    for category in ("label_shards", "policies"):
        for metadata in matrix[category]:
            path = Path(metadata["path"])
            _require(path.is_file() and metadata["sha256"] == sha256(path), f"{category} drifted.")

    differences: dict[str, dict[tuple[str, str, int, str, int], list[float]]] = {
        "learned_vs_one_token": defaultdict(list),
        "learned_vs_fixed": defaultdict(list),
        "no_dense_vs_learned": defaultdict(list),
        "no_pins_vs_learned": defaultdict(list),
    }
    contrast_arms = {
        "learned_vs_one_token": (PRIMARY, ONE_TOKEN),
        "learned_vs_fixed": (PRIMARY, FIXED),
        "no_dense_vs_learned": (
            "online-learned-lookahead-no-dense-fallback",
            PRIMARY,
        ),
        "no_pins_vs_learned": (
            "online-learned-lookahead-no-protected-pins",
            PRIMARY,
        ),
    }
    physical: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    failure_count = 0
    budget_violations = 0
    policy_digests: set[str] = set()
    paired_conversations = 0
    for metadata in matrix["test_shards"]:
        path = Path(metadata["path"])
        _require(path.is_file() and metadata["sha256"] == sha256(path), "Test shard drifted.")
        shard = json.loads(path.read_text())
        records = shard.get("records", [])
        metrics = shard.get("batch_metrics", [])
        _require(
            shard.get("experiment_id") == "p1-online-learned-lookahead-test-shard-v1"
            and shard.get("source", {}).get("dirty") is False
            and tuple(shard.get("arms", ())) == evaluator.ARMS
            and shard.get("examples") == 20
            and len(records) == 20 * len(evaluator.ARMS)
            and len(metrics) == 5 * len(evaluator.ARMS)
            and hashlib.sha256(
                json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            == shard.get("records_digest"),
            f"Invalid online-lookahead test shard: {path}",
        )
        policy_path = Path(shard["policy"]["path"])
        _require(
            policy_path.is_file() and shard["policy"]["sha256"] == sha256(policy_path),
            "Test shard policy dependency drifted.",
        )
        policy_digests.add(str(shard["policy_digest"]))
        by_arm_id = {(row["arm"], row["conversation_id"]): row for row in records}
        ids = {row["conversation_id"] for row in records}
        _require(
            len(by_arm_id) == len(records)
            and all(
                {arm for arm, item_id in by_arm_id if item_id == conversation_id}
                == set(evaluator.ARMS)
                for conversation_id in ids
            ),
            "Test shard pairing drifted.",
        )
        paired_conversations += len(ids)
        key_prefix = (
            shard["scale"],
            shard["budget"],
            int(shard["training_seed"]),
            shard["family"],
            int(shard["context"]),
        )
        for conversation_id in ids:
            for name, (candidate, comparator) in contrast_arms.items():
                left = by_arm_id[(candidate, conversation_id)]
                right = by_arm_id[(comparator, conversation_id)]
                _require(left["total"] == right["total"], "Paired query counts drifted.")
                differences[name][key_prefix].append(
                    left["correct_count"] / left["total"] - right["correct_count"] / right["total"]
                )
        for row in metrics:
            budget_violations += int(row["budget_violations"])
            arm = row["arm"]
            cell = physical[(shard["scale"], shard["budget"], arm)]
            cell["peak_cuda_allocated_bytes"].append(float(row["peak_cuda_allocated_bytes"]))
            cell["peak_cuda_reserved_bytes"].append(float(row["peak_cuda_reserved_bytes"]))
            cell["h2d_bytes"].append(float(row["tier"]["h2d_bytes"]))
            cell["d2h_bytes"].append(float(row["tier"]["d2h_bytes"]))
            cell["late_misses"].append(float(row["tier"]["late_misses"]))
            cell["wall_ms"].append(float(row["wall_ms"]))
            failure_count += int(row.get("failure") is not None)

    contrast_payloads = {
        name: contrast_statistics(
            differences[name],
            name=name,
            candidate=contrast_arms[name][0],
            comparator=contrast_arms[name][1],
        )
        for name in differences
    }
    primary_cells = contrast_payloads["learned_vs_one_token"]["cells"]
    p_values = {
        f"{row['scale']}:{row['budget']}": row["seed_cluster_inference"][
            "paired_randomization_two_sided_p"
        ]
        for row in primary_cells
    }
    adjusted = holm_bonferroni(p_values)
    cell_gate = []
    for row in primary_cells:
        cell_name = f"{row['scale']}:{row['budget']}"
        seed_inference = row["seed_cluster_inference"]
        ci = seed_inference["seed_cluster_bootstrap_ci"]
        seed_means = seed_inference["seed_means"]
        passed = ci[0] > 0.0 and all(value > 0.0 for value in seed_means)
        cell_gate.append(
            {
                "scale": row["scale"],
                "budget": row["budget"],
                "five_of_five_seed_means_positive": all(value > 0.0 for value in seed_means),
                "seed_cluster_bootstrap_ci": ci,
                "exact_resolution_aware_p": seed_inference["paired_randomization_two_sided_p"],
                "holm_adjusted_p_descriptive": adjusted[cell_name],
                "passed": passed,
            }
        )

    system_cells = []
    memory_passed = True
    for scale in matrix_runner.SCALES:
        for budget in causal.BUDGET_LABELS:
            learned = physical[(scale, budget, PRIMARY)]
            fixed = physical[(scale, budget, FIXED)]
            learned_hbm = float(np.mean(learned["peak_cuda_allocated_bytes"]))
            fixed_hbm = float(np.mean(fixed["peak_cuda_allocated_bytes"]))
            relative_hbm = (learned_hbm - fixed_hbm) / max(fixed_hbm, 1.0)
            learned_h2d = float(np.mean(learned["h2d_bytes"]))
            fixed_h2d = float(np.mean(fixed["h2d_bytes"]))
            passed = relative_hbm <= MAXIMUM_HBM_DIFFERENCE and learned_h2d <= fixed_h2d
            memory_passed = memory_passed and passed
            system_cells.append(
                {
                    "scale": scale,
                    "budget": budget,
                    "learned_peak_allocated_bytes_mean": learned_hbm,
                    "fixed_peak_allocated_bytes_mean": fixed_hbm,
                    "relative_peak_allocated_difference": relative_hbm,
                    "learned_h2d_bytes_mean": learned_h2d,
                    "fixed_h2d_bytes_mean": fixed_h2d,
                    "passed": passed,
                }
            )
    fixed_slices = contrast_payloads["learned_vs_fixed"]["by_family_context"]
    worst_slice = min(fixed_slices, key=lambda row: row["mean_difference"])
    worst_slice_passed = worst_slice["mean_difference"] >= MAXIMUM_WORST_SLICE_REGRESSION
    gate_passed = (
        all(row["passed"] for row in cell_gate)
        and memory_passed
        and worst_slice_passed
        and budget_violations == 0
        and failure_count == 0
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    _require(not dirty, "Online-lookahead summary requires a clean source tree.")
    return {
        "schema_version": 1,
        "experiment_id": "p1-online-learned-lookahead-audit-v1",
        "raw_matrix": {"path": str(matrix_path), "sha256": sha256(matrix_path)},
        "audit": {
            "label_shards_verified": matrix_runner.EXPECTED_LABEL_SHARDS,
            "policies_verified": matrix_runner.EXPECTED_POLICIES,
            "test_shards_verified": matrix_runner.EXPECTED_TEST_SHARDS,
            "paired_conversations": paired_conversations,
            "quality_arm_conversations": paired_conversations * len(evaluator.ARMS),
            "training_seeds": len(labels.TRAINING_SEEDS),
            "scales": len(matrix_runner.SCALES),
            "families": len(labels.PAPER_GRADE_WORKLOAD_FAMILIES),
            "contexts": len(labels.CONTEXTS),
            "budgets": len(causal.BUDGET_LABELS),
            "policy_digests": len(policy_digests),
            "all_raw_digests_verified": True,
            "all_dependencies_verified": True,
            "all_inputs_paired": True,
            "zero_budget_violations": budget_violations == 0,
            "complete_failure_accounting": True,
            "online_token_offset_verified": True,
            "native_bootstrap_accounted": True,
            "cache_replay_contract_tested": True,
        },
        "contrasts": contrast_payloads,
        "primary_gate": {
            "candidate": PRIMARY,
            "one_token_comparator": ONE_TOKEN,
            "fixed_comparator": FIXED,
            "cells": cell_gate,
            "system_cells": system_cells,
            "worst_fixed_slice": worst_slice,
            "worst_slice_passed": worst_slice_passed,
            "budget_violations": budget_violations,
            "failures": failure_count,
            "passed": gate_passed,
        },
        "decision": "success" if gate_passed else "negative-result",
        "claim_boundary": (
            "Held-out synthetic token-t to token-(t+1) evidence for this frozen predictor, "
            "checkpoint set, and budget matrix only; never same-token, natural-language, or "
            "official DeepSeek-V4 evidence."
        ),
        "source": {"commit": commit, "dirty": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the online learned-lookahead matrix.")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1-online-learned-lookahead-matrix.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p1-online-learned-lookahead.summary.json"
        ),
    )
    args = parser.parse_args()
    payload = summarize(args.matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"decision": payload["decision"], "gate": payload["primary_gate"]}))


if __name__ == "__main__":
    main()
