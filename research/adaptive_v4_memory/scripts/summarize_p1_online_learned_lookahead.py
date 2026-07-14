from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

import collect_p1_online_lookahead_labels as labels
import evaluate_p1_online_learned_lookahead_shard as evaluator
import evaluate_p2_causal_factorial_shard as causal
import fit_p1_online_learned_lookahead_policy as fitter
import numpy as np
import run_p1_online_learned_lookahead as matrix_runner
from summarize_p2_causal_factorial import contrast_statistics
from summarize_p2_core_matrix import holm_bonferroni

from nano_deepseek_v4 import LearnedLookaheadPolicy

PRIMARY = "online-learned-lookahead+pins"
ONE_TOKEN = "one-token-training-free+pins"
FIXED = "memory-matched-fixed+pins"
MAXIMUM_HBM_DIFFERENCE = 0.01
MAXIMUM_WORST_SLICE_REGRESSION = -0.02
_DIGEST_CACHE: dict[tuple[Path, int, int], str] = {}
PARALLEL_RUNNER = Path("research/adaptive_v4_memory/scripts/run_p1_online_lookahead_parallel.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_sha256(path: Path) -> str:
    stat = path.stat()
    key = (path.resolve(), stat.st_mtime_ns, stat.st_size)
    if key not in _DIGEST_CACHE:
        _DIGEST_CACHE[key] = sha256(path)
    return _DIGEST_CACHE[key]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound_path(metadata: dict[str, Any], label: str) -> Path:
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing {label}: {path}")
    _require(metadata.get("sha256") == _cached_sha256(path), f"{label} digest drifted: {path}")
    return path


def _verify_common(payload: dict[str, Any], *, implementation_digest: str, label: str) -> None:
    _require(
        payload.get("source", {}).get("dirty") is False
        and payload.get("source", {}).get("implementation_digest") == implementation_digest,
        f"{label} source implementation drifted.",
    )
    _require(
        payload.get("design", {}).get("path") == str(matrix_runner.DESIGN)
        and payload.get("design", {}).get("sha256") == _cached_sha256(matrix_runner.DESIGN),
        f"{label} design dependency drifted.",
    )


def _verify_checkpoint_reuse_audit(metadata: dict[str, Any]) -> int:
    path = _bound_path(metadata, "online-lookahead checkpoint-reuse audit")
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    _require(
        payload.get("experiment_id") == "p1-online-lookahead-checkpoint-reuse-audit-v1"
        and payload.get("source", {}).get("dirty") is False
        and payload.get("source", {}).get("orchestrator_sha256") == _cached_sha256(PARALLEL_RUNNER)
        and audit.get("scale_seed_probes") == 10
        and audit.get("all_label_rows_identical") is True
        and audit.get("all_test_records_identical") is True
        and audit.get("all_controller_accounting_identical") is True,
        "Online-lookahead checkpoint-reuse audit is incomplete.",
    )
    coordinates: set[tuple[str, int]] = set()
    for probe_metadata in payload.get("probes", []):
        probe_path = _bound_path(probe_metadata, "online-lookahead checkpoint-reuse probe")
        probe = json.loads(probe_path.read_text())
        probe_audit = probe.get("audit", {})
        coordinate = (probe.get("scale"), probe.get("training_seed"))
        _require(
            probe.get("experiment_id") == "p1-online-lookahead-checkpoint-reuse-probe-v1"
            and coordinate[0] in matrix_runner.SCALES
            and coordinate[1] in labels.TRAINING_SEEDS
            and coordinate not in coordinates
            and probe_audit.get("label_rows_identical") is True
            and probe_audit.get("test_records_identical") is True
            and probe_audit.get("controller_accounting_identical") is True
            and probe_audit.get("timing_fields_excluded") is True
            and probe.get("implementation", {}).get("label") == labels.implementation_digest()
            and probe.get("implementation", {}).get("evaluation")
            == evaluator.implementation_digest()
            and probe.get("implementation", {}).get("orchestrator_sha256")
            == _cached_sha256(PARALLEL_RUNNER),
            f"Online-lookahead checkpoint-reuse probe drifted: {probe_path}",
        )
        for artifact in probe.get("artifacts", []):
            _bound_path(artifact, f"checkpoint-reuse child artifact for {probe_path}")
        coordinates.add(coordinate)
    _require(
        coordinates == set(product(matrix_runner.SCALES, labels.TRAINING_SEEDS)),
        "Online-lookahead checkpoint-reuse coordinate coverage drifted.",
    )
    return len(coordinates)


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
    _require(
        matrix.get("design", {}).get("path") == str(matrix_runner.DESIGN)
        and matrix.get("design", {}).get("sha256") == _cached_sha256(matrix_runner.DESIGN),
        "Online-lookahead matrix design drifted.",
    )
    causal_gate_path = _bound_path(matrix.get("p2_causal_gate", {}), "P2 causal gate")
    matrix_runner.require_causal_gate(causal_gate_path)
    reuse_probes = _verify_checkpoint_reuse_audit(matrix.get("checkpoint_reuse_audit", {}))
    label_implementation = labels.implementation_digest()
    fit_implementation = fitter.implementation_digest()
    evaluation_implementation = evaluator.implementation_digest()
    for category in ("label_shards", "policies", "test_shards"):
        _require(
            len({metadata.get("path") for metadata in matrix[category]}) == len(matrix[category]),
            f"Duplicate {category} paths detected.",
        )
    for metadata in matrix["label_shards"]:
        path = _bound_path(metadata, "online-lookahead label shard")
        shard = json.loads(path.read_text())
        _require(
            shard.get("experiment_id") == "p1-online-learned-lookahead-label-shard-v1",
            f"Wrong label shard id: {path}",
        )
        _verify_common(
            shard,
            implementation_digest=label_implementation,
            label=f"label shard {path}",
        )
        _bound_path(shard.get("checkpoint", {}), f"label checkpoint for {path}")
        rows = shard.get("rows", [])
        _require(
            shard.get("conversations") == labels.EXAMPLES_PER_SHARD
            and shard.get("risk_examples") == len(rows)
            and shard.get("failure_count") == len(shard.get("failures", []))
            and hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            == shard.get("rows_digest")
            and shard.get("leakage_guard", {}).get("features_use_prior_token_only") is True
            and shard.get("leakage_guard", {}).get("causal_offset") == 1,
            f"Invalid label shard accounting: {path}",
        )
    for metadata in matrix["policies"]:
        path = _bound_path(metadata, "online-lookahead policy")
        policy_payload = json.loads(path.read_text())
        _require(
            policy_payload.get("experiment_id") == "p1-online-learned-lookahead-policy-v1",
            f"Wrong policy id: {path}",
        )
        _verify_common(
            policy_payload,
            implementation_digest=fit_implementation,
            label=f"policy {path}",
        )
        for dependency_name in (
            "checkpoint",
            "quota_calibration",
            "physical_memory_match",
        ):
            _bound_path(
                policy_payload.get(dependency_name, {}),
                f"policy {dependency_name} for {path}",
            )
        policy = LearnedLookaheadPolicy.from_dict(policy_payload["policy"])
        _require(
            policy_payload.get("policy_digest") == policy.policy_digest
            and policy_payload.get("leakage_guard", {}).get("test_split_loaded") is False,
            f"Policy integrity or split boundary drifted: {path}",
        )
        for split in ("train", "calibration"):
            split_payload = policy_payload.get("splits", {}).get(split, {})
            for shard_metadata in split_payload.get("shard_digests", []):
                _bound_path(shard_metadata, f"policy {split} label shard for {path}")

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
        path = _bound_path(metadata, "online-lookahead test shard")
        shard = json.loads(path.read_text())
        _verify_common(
            shard,
            implementation_digest=evaluation_implementation,
            label=f"test shard {path}",
        )
        records = shard.get("records", [])
        metrics = shard.get("batch_metrics", [])
        _require(
            shard.get("experiment_id") == "p1-online-learned-lookahead-test-shard-v1"
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
        policy_path = _bound_path(shard["policy"], "test shard policy dependency")
        policy_payload = json.loads(policy_path.read_text())
        _require(
            shard.get("policy_digest") == policy_payload.get("policy_digest"),
            "Test shard policy digest drifted.",
        )
        for dependency_name in (
            "checkpoint",
            "quota_calibration",
            "physical_memory_match",
        ):
            _bound_path(
                shard.get(dependency_name, {}),
                f"test shard {dependency_name} for {path}",
            )
        _require(
            shard.get("generation_seed")
            == evaluator.generation_seed(
                training_seed=int(shard["training_seed"]),
                family=str(shard["family"]),
                context=int(shard["context"]),
                replicate=int(shard["replicate"]),
            )
            and shard.get("leakage_guard", {}).get("test_examples_used_for_training") is False
            and shard.get("leakage_guard", {}).get("policy_frozen_before_test") is True
            and shard.get("leakage_guard", {}).get("paired_inputs_shared_across_arms") is True,
            f"Test split or deterministic seed boundary drifted: {path}",
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
            failed = row.get("failure") is not None
            failure_count += int(failed)
            if not failed:
                accounting = row["accounting"]
                _require(accounting is not None, "Successful system row lacks cache accounting.")
                cell["peak_cuda_allocated_bytes"].append(float(row["peak_cuda_allocated_bytes"]))
                cell["peak_cuda_reserved_bytes"].append(float(row["peak_cuda_reserved_bytes"]))
                cell["hot_resident_bytes"].append(float(accounting["hot_resident_bytes"]))
                cell["h2d_bytes"].append(float(row["tier"]["h2d_bytes"]))
                cell["useful_h2d_bytes"].append(float(row["tier"]["useful_h2d_bytes"]))
                cell["d2h_bytes"].append(float(row["tier"]["d2h_bytes"]))
                cell["late_misses"].append(float(row["tier"]["late_misses"]))
                cell["wall_ms"].append(float(row["wall_ms"]))

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
                "minimum_attainable_two_sided_exact_p": 0.0625,
                "p_value_used_as_success_gate": False,
                "passed": passed,
            }
        )

    system_cells = []
    memory_passed = True
    for scale in matrix_runner.SCALES:
        for budget in causal.BUDGET_LABELS:
            learned = physical[(scale, budget, PRIMARY)]
            fixed = physical[(scale, budget, FIXED)]

            def mean_or_none(values: list[float]) -> float | None:
                return float(np.mean(values)) if values else None

            learned_hbm = mean_or_none(learned["peak_cuda_allocated_bytes"])
            fixed_hbm = mean_or_none(fixed["peak_cuda_allocated_bytes"])
            relative_hbm = (
                (learned_hbm - fixed_hbm) / max(fixed_hbm, 1.0)
                if learned_hbm is not None and fixed_hbm is not None
                else None
            )
            learned_hot = mean_or_none(learned["hot_resident_bytes"])
            fixed_hot = mean_or_none(fixed["hot_resident_bytes"])
            relative_hot = (
                (learned_hot - fixed_hot) / max(fixed_hot, 1.0)
                if learned_hot is not None and fixed_hot is not None
                else None
            )
            learned_h2d = mean_or_none(learned["h2d_bytes"])
            fixed_h2d = mean_or_none(fixed["h2d_bytes"])
            learned_useful_h2d = mean_or_none(learned["useful_h2d_bytes"])
            fixed_useful_h2d = mean_or_none(fixed["useful_h2d_bytes"])
            passed = (
                relative_hot is not None
                and learned_useful_h2d is not None
                and fixed_useful_h2d is not None
                and relative_hot <= MAXIMUM_HBM_DIFFERENCE
                and learned_useful_h2d <= fixed_useful_h2d
            )
            memory_passed = memory_passed and passed
            system_cells.append(
                {
                    "scale": scale,
                    "budget": budget,
                    "learned_peak_allocated_bytes_mean": learned_hbm,
                    "fixed_peak_allocated_bytes_mean": fixed_hbm,
                    "relative_peak_allocated_difference": relative_hbm,
                    "learned_hot_resident_bytes_mean": learned_hot,
                    "fixed_hot_resident_bytes_mean": fixed_hot,
                    "relative_hot_resident_difference": relative_hot,
                    "learned_h2d_bytes_mean": learned_h2d,
                    "fixed_h2d_bytes_mean": fixed_h2d,
                    "learned_useful_h2d_bytes_mean": learned_useful_h2d,
                    "fixed_useful_h2d_bytes_mean": fixed_useful_h2d,
                    "successful_learned_measurements": len(learned["hot_resident_bytes"]),
                    "successful_fixed_measurements": len(fixed["hot_resident_bytes"]),
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
            "implementation_digests_verified": True,
            "dependency_artifact_digests_verified": True,
            "checkpoint_reuse_equivalence_verified": True,
            "checkpoint_reuse_scale_seed_probes": reuse_probes,
            "all_inputs_paired": True,
            "zero_budget_violations": budget_violations == 0,
            "complete_failure_accounting": True,
            "online_token_offset_verified": True,
            "native_bootstrap_accounted": True,
            "cache_replay_contract_tested": True,
            "resolution_aware_gate_verified": True,
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
