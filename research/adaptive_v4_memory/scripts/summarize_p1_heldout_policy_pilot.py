from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from nano_deepseek_v4 import PAPER_GRADE_WORKLOAD_FAMILIES

EXPECTED_CONTEXTS = (80, 128, 256, 512, 1024)
EXPECTED_POLICIES = (
    "native",
    "fixed-1x",
    "fixed-2x",
    "fixed-4x",
    "calibrated-local-1x",
    "calibrated-local-2x",
    "calibrated-local-4x",
    "calibrated-hierarchical-1x",
    "calibrated-hierarchical-2x",
    "calibrated-hierarchical-4x",
    "calibrated-hierarchical-fallback-1x",
    "calibrated-hierarchical-fallback-2x",
    "calibrated-hierarchical-fallback-4x",
)
ANALYSIS_SEED = 9071401
RESAMPLES = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _record_digest(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _comparison_seed(candidate: str, comparator: str) -> int:
    digest = hashlib.sha256(f"{ANALYSIS_SEED}:{candidate}:{comparator}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_comparison(
    by_policy: dict[str, dict[tuple[str, int, str], dict[str, Any]]],
    *,
    candidate: str,
    comparator: str,
) -> dict[str, Any]:
    candidate_rows = by_policy[candidate]
    comparator_rows = by_policy[comparator]
    _require(candidate_rows.keys() == comparator_rows.keys(), "Paired key drift.")
    keys = sorted(candidate_rows)
    pairs: list[tuple[int, int, int]] = []
    recovered = 0
    regressed = 0
    identical_predictions = True
    for key in keys:
        left = candidate_rows[key]
        right = comparator_rows[key]
        _require(left["targets"] == right["targets"], f"Target drift at {key}.")
        _require(left["query_positions"] == right["query_positions"], f"Query drift at {key}.")
        _require(
            left["evidence_positions"] == right["evidence_positions"],
            f"Evidence drift at {key}.",
        )
        pairs.append((left["correct_count"], right["correct_count"], left["total"]))
        recovered += sum(
            not old and new
            for new, old in zip(left["correct"], right["correct"], strict=True)
        )
        regressed += sum(
            old and not new
            for new, old in zip(left["correct"], right["correct"], strict=True)
        )
        identical_predictions &= left["predictions"] == right["predictions"]

    candidate_correct = sum(left for left, _, _ in pairs)
    comparator_correct = sum(right for _, right, _ in pairs)
    total = sum(count for _, _, count in pairs)
    observed = (candidate_correct - comparator_correct) / total
    rng = random.Random(_comparison_seed(candidate, comparator))
    bootstrap = []
    for _ in range(RESAMPLES):
        sampled = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        bootstrap.append(
            sum(left - right for left, right, _ in sampled)
            / sum(count for _, _, count in sampled)
        )
    more_extreme = 0
    for _ in range(RESAMPLES):
        permuted = sum(
            (left - right) * (-1 if rng.random() < 0.5 else 1)
            for left, right, _ in pairs
        ) / total
        more_extreme += abs(permuted) >= abs(observed)
    candidate_h2d = statistics.fmean(float(candidate_rows[key]["h2d_bytes"]) for key in keys)
    comparator_h2d = statistics.fmean(float(comparator_rows[key]["h2d_bytes"]) for key in keys)
    return {
        "candidate": candidate,
        "comparator": comparator,
        "conversations": len(pairs),
        "queries": total,
        "candidate_correct": candidate_correct,
        "comparator_correct": comparator_correct,
        "accuracy_difference": observed,
        "accuracy_difference_pp": observed * 100.0,
        "paired_cluster_bootstrap_95_ci": [
            _quantile(bootstrap, 0.025),
            _quantile(bootstrap, 0.975),
        ],
        "paired_sign_flip_p_value": (more_extreme + 1) / (RESAMPLES + 1),
        "query_recoveries": recovered,
        "query_regressions": regressed,
        "predictions_identical": identical_predictions,
        "mean_h2d_bytes_candidate": candidate_h2d,
        "mean_h2d_bytes_comparator": comparator_h2d,
        "mean_h2d_ratio": candidate_h2d / comparator_h2d if comparator_h2d else None,
    }


def _aggregate(records: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in fields)].append(record)
    result = []
    for key, rows in sorted(groups.items()):
        correct = sum(row["correct_count"] for row in rows)
        total = sum(row["total"] for row in rows)
        item = {field: value for field, value in zip(fields, key, strict=True)}
        item.update(
            {
                "conversations": len(rows),
                "correct": correct,
                "total": total,
                "accuracy": correct / total,
                "mean_hot_resident_bytes": statistics.fmean(
                    float(row["hot_resident_bytes"]) for row in rows
                ),
                "mean_h2d_bytes": statistics.fmean(float(row["h2d_bytes"]) for row in rows),
                "mean_d2h_bytes": statistics.fmean(float(row["d2h_bytes"]) for row in rows),
            }
        )
        result.append(item)
    return result


def _audit_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records", [])
    _require(isinstance(records, list) and bool(records), "No held-out records.")
    _require(_record_digest(records) == payload.get("records_digest"), "Record digest drift.")
    examples = payload["examples_per_family_policy"]
    expected_per_policy = examples * len(PAPER_GRADE_WORKLOAD_FAMILIES)
    expected_total = expected_per_policy * len(EXPECTED_POLICIES)
    _require(len(records) == expected_total, "Held-out record count drift.")
    by_policy_count: dict[str, int] = defaultdict(int)
    by_slice_count: dict[tuple[str, str, int], int] = defaultdict(int)
    for record in records:
        policy = record["policy"]
        _require(policy in EXPECTED_POLICIES, f"Unknown policy: {policy}")
        _require(record["family"] in PAPER_GRADE_WORKLOAD_FAMILIES, "Family drift.")
        _require(record["context"] in EXPECTED_CONTEXTS, "Context drift.")
        _require(record["budget_violations"] == 0, "Controller budget violation recorded.")
        predictions = record["predictions"]
        targets = record["targets"]
        correctness = [left == right for left, right in zip(predictions, targets, strict=True)]
        _require(correctness == record["correct"], "Correctness field drift.")
        _require(sum(correctness) == record["correct_count"], "Correct count drift.")
        _require(len(correctness) == record["total"], "Query total drift.")
        by_policy_count[policy] += 1
        by_slice_count[(policy, record["family"], record["context"])] += 1
    _require(
        set(by_policy_count) == set(EXPECTED_POLICIES)
        and all(count == expected_per_policy for count in by_policy_count.values()),
        "Policy coverage drift.",
    )
    expected_per_slice = examples // len(EXPECTED_CONTEXTS)
    _require(
        len(by_slice_count)
        == len(EXPECTED_POLICIES) * len(PAPER_GRADE_WORKLOAD_FAMILIES) * len(EXPECTED_CONTEXTS),
        "Policy/family/context coverage drift.",
    )
    _require(
        all(count == expected_per_slice for count in by_slice_count.values()),
        "Unbalanced policy/family/context slices.",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize a P1 held-out pilot.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    _require(
        payload.get("experiment_id") == "p1-heldout-policy-pilot-v1",
        "Wrong held-out experiment id.",
    )
    _require(payload.get("source", {}).get("dirty") is False, "Held-out run used dirty source.")
    _require(tuple(payload.get("families", ())) == PAPER_GRADE_WORKLOAD_FAMILIES, "Family drift.")
    _require(tuple(payload.get("contexts", ())) == EXPECTED_CONTEXTS, "Context drift.")
    _require(tuple(payload.get("policies", ())) == EXPECTED_POLICIES, "Policy order drift.")
    _require(payload.get("evaluation_seed_namespace") == "held_out_evaluation", "Seed drift.")
    _require(payload.get("evaluation_seed") == 8071401, "Pilot evaluation seed drift.")
    checkpoint = payload["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    _require(checkpoint_path.is_file(), "Missing held-out checkpoint.")
    _require(checkpoint_path.stat().st_size == checkpoint["bytes"], "Checkpoint size drift.")
    _require(_sha256(checkpoint_path) == checkpoint["sha256"], "Checkpoint digest drift.")
    calibration = payload["calibration_artifact"]
    calibration_path = Path(calibration["path"])
    _require(calibration_path.is_file(), "Missing calibration artifact.")
    _require(_sha256(calibration_path) == calibration["sha256"], "Calibration digest drift.")
    _require(calibration["seed"] != payload["evaluation_seed"], "Calibration/evaluation leakage.")
    records = _audit_records(payload)

    by_policy: dict[str, dict[tuple[str, int, str], dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (record["family"], record["context"], record["conversation_id"])
        _require(key not in by_policy[record["policy"]], f"Duplicate paired key: {key}")
        by_policy[record["policy"]][key] = record
    reference_keys = by_policy["native"].keys()
    _require(
        all(rows.keys() == reference_keys for rows in by_policy.values()),
        "Policies did not share exact paired examples.",
    )

    comparison_pairs: list[tuple[str, str]] = []
    for multiplier in (1, 2, 4):
        fixed = f"fixed-{multiplier}x"
        comparison_pairs.extend(
            (
                (f"calibrated-local-{multiplier}x", fixed),
                (f"calibrated-hierarchical-{multiplier}x", fixed),
                (
                    f"calibrated-hierarchical-fallback-{multiplier}x",
                    f"calibrated-hierarchical-{multiplier}x",
                ),
                (
                    f"calibrated-hierarchical-{multiplier}x",
                    f"calibrated-local-{multiplier}x",
                ),
            )
        )
    comparisons = [
        _paired_comparison(by_policy, candidate=candidate, comparator=comparator)
        for candidate, comparator in comparison_pairs
    ]
    overall = _aggregate(records, ("policy",))
    family = _aggregate(records, ("policy", "family"))
    slices = _aggregate(records, ("policy", "family", "context"))
    output = {
        "schema_version": 1,
        "experiment_id": "p1-heldout-policy-pilot-audit-v1",
        "interpretation": "single-checkpoint directional held-out pilot; not a paper-grade claim",
        "raw_artifact": {"path": str(args.input), "sha256": _sha256(args.input)},
        "source": payload["source"],
        "checkpoint": checkpoint,
        "calibration_artifact": calibration,
        "design": {
            "scale": payload["scale"],
            "evaluation_seed": payload["evaluation_seed"],
            "families": PAPER_GRADE_WORKLOAD_FAMILIES,
            "contexts": EXPECTED_CONTEXTS,
            "policies": EXPECTED_POLICIES,
            "examples_per_family_policy": payload["examples_per_family_policy"],
            "conversations_per_policy": len(reference_keys),
            "policy_conversation_records": len(records),
            "analysis_seed": ANALYSIS_SEED,
            "paired_resamples": RESAMPLES,
        },
        "validation": {
            "clean_source_verified": True,
            "raw_record_digest_verified": True,
            "checkpoint_digest_verified": True,
            "calibration_digest_verified": True,
            "exact_paired_examples_verified": True,
            "balanced_family_context_grid_verified": True,
            "all_correctness_recomputed": True,
            "zero_budget_violations": True,
            "calibration_evaluation_seed_isolation_verified": True,
        },
        "overall": overall,
        "by_family": family,
        "by_family_context": slices,
        "paired_comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"design": output["design"], "validation": output["validation"]}))


if __name__ == "__main__":
    main()
