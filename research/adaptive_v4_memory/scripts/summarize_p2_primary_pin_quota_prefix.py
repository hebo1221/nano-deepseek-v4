from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, cast

import run_p2_primary_pin_quota_matrix as matrix

PREFIX_CELLS = 4_950
CONTRASTS = {
    "adaptive_quota_with_pins": ("calibrated+pins", "fixed+pins"),
    "adaptive_quota_without_pins": ("calibrated-no-pins", "fixed"),
    "pins_under_fixed_quota": ("fixed+pins", "fixed"),
    "pins_under_calibrated_quota": ("calibrated+pins", "calibrated-no-pins"),
}
SYSTEM_METRICS = (
    "h2d_bytes",
    "d2h_bytes",
    "peak_allocated_bytes",
    "peak_reserved_bytes",
)


class Totals:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.positive = 0
        self.zero = 0
        self.negative = 0

    def add(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError("Non-finite postmortem observation.")
        self.count += 1
        self.total += value
        self.positive += value > 0.0
        self.zero += value == 0.0
        self.negative += value < 0.0

    @property
    def mean(self) -> float:
        if not self.count:
            raise ValueError("Empty postmortem group.")
        return self.total / self.count


def _key(coordinate: dict[str, Any]) -> tuple[Any, ...]:
    return matrix._coordinate_key(coordinate)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def seed_inference(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Seed inference requires observations.")
    observed = sum(values) / len(values)
    resampled = [
        sum(values[index] for index in indices) / len(values)
        for indices in product(range(len(values)), repeat=len(values))
    ]
    signed = [
        sum(value * sign for value, sign in zip(values, signs, strict=True)) / len(values)
        for signs in product((-1.0, 1.0), repeat=len(values))
    ]
    return {
        "independent_training_seeds": len(values),
        "mean_difference": observed,
        "mean_difference_percentage_points": observed * 100.0,
        "seed_bootstrap_95_ci": [_quantile(resampled, 0.025), _quantile(resampled, 0.975)],
        "exact_sign_flip_two_sided_p": sum(
            abs(value) >= abs(observed) - 1e-15 for value in signed
        )
        / len(signed),
        "positive_seeds": sum(value > 0.0 for value in values),
        "zero_seeds": sum(value == 0.0 for value in values),
        "negative_seeds": sum(value < 0.0 for value in values),
        "seed_values": values,
    }


class Analysis:
    def __init__(self) -> None:
        self.arm_accuracy: defaultdict[tuple[str, str, int, str], Totals] = defaultdict(Totals)
        self.effects: defaultdict[tuple[str, str, int, str], Totals] = defaultdict(Totals)
        self.family_effects: defaultdict[tuple[str, str, str, str], Totals] = defaultdict(Totals)
        self.context_effects: defaultdict[tuple[str, str, int, str], Totals] = defaultdict(Totals)
        self.system_effects: defaultdict[tuple[str, str, int, str, str], Totals] = defaultdict(
            Totals
        )

    def add(self, payload: dict[str, Any]) -> None:
        coordinate = payload["coordinate"]
        scale = cast(str, coordinate["scale"])
        budget = cast(str, coordinate["budget"])
        seed = cast(int, coordinate["training_seed"])
        family = cast(str, coordinate["family"])
        context = cast(int, coordinate["context"])
        examples = {row["example_index"]: row for row in payload["examples"]}
        outcomes = {(row["example_index"], row["arm"]): row for row in payload["outcomes"]}
        for example_index, example in examples.items():
            targets = example["targets"]
            for arm in matrix.ARMS:
                row = outcomes[(example_index, arm)]
                predictions = row["predictions"]
                if len(predictions) != len(targets):
                    raise ValueError("Prediction/target cardinality drifted.")
                for prediction, target in zip(predictions, targets, strict=True):
                    self.arm_accuracy[(scale, budget, seed, arm)].add(prediction == target)
            for name, (candidate, comparator) in CONTRASTS.items():
                candidate_row = outcomes[(example_index, candidate)]
                comparator_row = outcomes[(example_index, comparator)]
                for candidate_prediction, comparator_prediction, target in zip(
                    candidate_row["predictions"],
                    comparator_row["predictions"],
                    targets,
                    strict=True,
                ):
                    difference = float(candidate_prediction == target) - float(
                        comparator_prediction == target
                    )
                    self.effects[(scale, budget, seed, name)].add(difference)
                    self.family_effects[(scale, budget, family, name)].add(difference)
                    self.context_effects[(scale, budget, context, name)].add(difference)
                for metric in SYSTEM_METRICS:
                    self.system_effects[(scale, budget, seed, name, metric)].add(
                        float(candidate_row["token_summary"][metric])
                        - float(comparator_row["token_summary"][metric])
                    )

    def result(self) -> dict[str, Any]:
        available_cells = sorted({key[:2] for key in self.effects})
        contrasts: dict[str, Any] = {}
        for name, (candidate, comparator) in CONTRASTS.items():
            by_seed = [
                {
                    "scale": scale,
                    "budget": budget,
                    "training_seed": seed,
                    "paired_queries": totals.count,
                    "mean_difference": totals.mean,
                    "mean_difference_percentage_points": totals.mean * 100.0,
                    "candidate_only_correct": totals.positive,
                    "comparator_only_correct": totals.negative,
                    "ties": totals.zero,
                }
                for (scale, budget, seed, item_name), totals in sorted(self.effects.items())
                if item_name == name
            ]
            cells = []
            for scale, budget in available_cells:
                values: list[float] = [
                    cast(float, row["mean_difference"])
                    for row in by_seed
                    if row["scale"] == scale and row["budget"] == budget
                ]
                if values:
                    cells.append({"scale": scale, "budget": budget, **seed_inference(values)})
            families = [
                {
                    "scale": scale,
                    "budget": budget,
                    "family": family,
                    "paired_queries": totals.count,
                    "mean_difference": totals.mean,
                    "mean_difference_percentage_points": totals.mean * 100.0,
                    "candidate_only_correct": totals.positive,
                    "comparator_only_correct": totals.negative,
                    "ties": totals.zero,
                }
                for (scale, budget, family, item_name), totals in sorted(
                    self.family_effects.items()
                )
                if item_name == name
            ]
            contexts = [
                {
                    "scale": scale,
                    "budget": budget,
                    "context": context,
                    "paired_queries": totals.count,
                    "mean_difference": totals.mean,
                    "mean_difference_percentage_points": totals.mean * 100.0,
                    "candidate_only_correct": totals.positive,
                    "comparator_only_correct": totals.negative,
                    "ties": totals.zero,
                }
                for (scale, budget, context, item_name), totals in sorted(
                    self.context_effects.items()
                )
                if item_name == name
            ]
            systems = [
                {
                    "scale": scale,
                    "budget": budget,
                    "training_seed": seed,
                    "metric": metric,
                    "paired_examples": totals.count,
                    "mean_difference": totals.mean,
                }
                for (scale, budget, seed, item_name, metric), totals in sorted(
                    self.system_effects.items()
                )
                if item_name == name
            ]
            contrasts[name] = {
                "candidate": candidate,
                "comparator": comparator,
                "cells": cells,
                "by_seed": by_seed,
                "by_family": families,
                "by_context": contexts,
                "system_differences_by_seed": systems,
            }
        accuracies = [
            {
                "scale": scale,
                "budget": budget,
                "training_seed": seed,
                "arm": arm,
                "queries": totals.count,
                "accuracy": totals.mean,
            }
            for (scale, budget, seed, arm), totals in sorted(self.arm_accuracy.items())
        ]
        return {"arm_accuracy_by_seed": accuracies, "contrasts": contrasts}


def summarize(roots: dict[str, Path], *, prefix_cells: int = PREFIX_CELLS) -> dict[str, Any]:
    full = matrix.coordinates()
    if prefix_cells <= 0 or prefix_cells >= len(full):
        raise ValueError("Prefix boundary must be inside the frozen matrix.")
    expected_global = full[:prefix_cells]
    expected_keys = {_key(coordinate) for coordinate in expected_global}
    assignment = {
        site: [coordinate for coordinate in matrix.coordinates(site) if _key(coordinate) in expected_keys]
        for site in roots
    }
    if set(roots) != {"gb10", "rtx4090"}:
        raise ValueError("Both frozen mixed-device sites are required.")
    analysis = Analysis()
    observed_keys: set[tuple[Any, ...]] = set()
    site_counts: dict[str, int] = {}
    for site, root in roots.items():
        expected = assignment[site]
        expected_paths = {matrix.cell_path(root, coordinate) for coordinate in expected}
        observed_paths = set(root.rglob("replicate-*.json")) if root.is_dir() else set()
        missing = expected_paths - observed_paths
        unexpected = observed_paths - expected_paths
        if missing or unexpected:
            raise RuntimeError(
                f"{site} prefix is not closed: missing={len(missing)}, unexpected={len(unexpected)}"
            )
        for coordinate in expected:
            path = matrix.cell_path(root, coordinate)
            if path.is_symlink():
                raise RuntimeError(f"Symlinked result cell: {path}")
            payload = json.loads(path.read_text())
            experiment_id = payload.get("experiment_id")
            if experiment_id not in {matrix.EXPERIMENT_ID, matrix.LEGACY_EXPERIMENT_ID}:
                raise RuntimeError(f"Unknown result lineage: {path}")
            matrix.validate_cell(payload, coordinate, expected_experiment_id=experiment_id)
            coordinate_key = _key(coordinate)
            if coordinate_key in observed_keys:
                raise RuntimeError(f"Duplicate mixed-device coordinate: {coordinate}")
            observed_keys.add(coordinate_key)
            analysis.add(payload)
        site_counts[site] = len(expected)
    if observed_keys != expected_keys:
        raise RuntimeError("Combined mixed-device prefix coverage drifted.")
    body = {
        "schema_version": 1,
        "experiment_id": "p2-primary-pin-quota-4950-futility-postmortem-v1",
        "status": "exploratory_futility_truncated",
        "claim_boundary": (
            "Outcome-informed 4,950-cell prefix; descriptive root-cause evidence only, not the "
            "pre-registered 9,000-cell confirmatory gate."
        ),
        "coverage": {
            "global_cells": prefix_cells,
            "site_cells": site_counts,
            "included": "all s55 seeds at 2x/4x plus s151 seed-6071406 at 2x",
            "first_excluded_coordinate": full[prefix_cells],
        },
        "integrity": {
            "all_cells_payload_validated": True,
            "closed_prefix_verified": True,
            "duplicate_coordinates": 0,
            "future_cells_present": False,
        },
        "confirmatory_gate": {
            "status": "not_evaluable",
            "reason": "s151 lacks four seeds at 2x and all five seeds at 4x after futility stop",
        },
        **analysis.result(),
    }
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and summarize the closed P2 prefix.")
    parser.add_argument("--gb10-root", type=Path, default=matrix.output_root("gb10"))
    parser.add_argument("--rtx4090-root", type=Path, default=matrix.output_root("rtx4090"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize({"gb10": args.gb10_root, "rtx4090": args.rtx4090_root})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
