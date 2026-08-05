from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import run_causal_identifiability_atlas as runner


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty collection.")
    return math.fsum(values) / len(values)


def _flat(values: list[list[Any]]) -> list[Any]:
    return [item for row in values for item in row]


def _outcome_metric(cell: dict[str, Any], arm: str, metric: str) -> float:
    values = _flat(cell["outcomes"][arm][metric])
    if metric == "correct":
        return _mean([float(bool(value)) for value in values])
    return _mean([float(value) for value in values])


def _delta(cell: dict[str, Any], arm: str, metric: str) -> float:
    return _outcome_metric(cell, arm, metric) - _outcome_metric(cell, "native", metric)


def _rankdata(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two.")
    left_ranks = _rankdata(left)
    right_ranks = _rankdata(right)
    if float(left_ranks.std()) == 0.0 or float(right_ranks.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _bootstrap_mean(
    values: list[float], *, draws: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(draws, len(array)))
    estimates = array[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def _bootstrap_ratio(
    numerator: list[float],
    denominator: list[float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    if top.shape != bottom.shape or top.ndim != 1:
        raise ValueError("Bootstrap ratio inputs must be paired vectors.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(top), size=(draws, len(top)))
    sampled_denominator = bottom[indices].mean(axis=1)
    estimates = np.divide(
        top[indices].mean(axis=1),
        sampled_denominator,
        out=np.full(draws, -np.inf),
        where=sampled_denominator > 0.0,
    )
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def _validate_cell(
    cell: dict[str, Any],
    *,
    coordinate: dict[str, Any],
    manifest_sha256: str,
) -> None:
    if (
        cell.get("schema_version") != 1
        or cell.get("experiment_id") != runner.EXPERIMENT_ID
        or cell.get("manifest_sha256") != manifest_sha256
        or cell.get("coordinate") != coordinate
    ):
        raise ValueError("Atlas cell binding drifted.")
    if cell.get("workload_identity") != runner.workload_identity(coordinate):
        raise ValueError("Atlas workload identity drifted.")
    if cell.get("evaluation_seed") != runner.evaluation_seed(
        coordinate, _MANIFEST["evaluation_seed_namespace"]
    ):
        raise ValueError("Atlas evaluation seed drifted.")
    if set(cell.get("outcomes", {})) != set(runner.ARMS):
        raise ValueError("Atlas outcome arm coverage drifted.")
    if cell["outcomes"]["identity-replay"] != cell["outcomes"]["native"]:
        raise ValueError("Stored identity outcome differs from native.")
    route = cell["route_contract"]
    if (
        route.get("identity_replay_bitwise_equal") is not True
        or route.get("observable_routes_exclude_evidence_and_targets") is not True
    ):
        raise ValueError("Atlas route invariant is absent.")
    if runner._digest(route["observable_receipts"]) != route["observable_receipts_sha256"]:
        raise ValueError("Observable route receipt digest drifted.")
    if runner._digest(route["oracle_receipts"]) != route["oracle_receipts_sha256"]:
        raise ValueError("Oracle route receipt digest drifted.")
    topk = int(route["topk"])
    for receipt in route["observable_receipts"]:
        selected = receipt["selected_block_ends"]
        if set(selected) != {"native", *runner.OBSERVABLE_POLICIES}:
            raise ValueError("Observable policy receipt coverage drifted.")
        if any(len(ends) != topk or len(set(ends)) != topk for ends in selected.values()):
            raise ValueError("Observable route cardinality drifted.")
    for receipt in route["oracle_receipts"]:
        if len(receipt["selected_block_ends"]) != topk:
            raise ValueError("Evidence-oracle route cardinality drifted.")
    atlas = cell["exhaustive_atlas"]
    if runner.is_exhaustive_coordinate(coordinate):
        if (
            atlas is None
            or atlas.get("fixed_batch_shape_with_discarded_repeat_last_padding") is not True
            or len(atlas["layers"]) != 5
        ):
            raise ValueError("Exhaustive s151 cell lacks five CSA layer atlases.")
        if runner._digest(atlas["layers"]) != atlas["layers_sha256"]:
            raise ValueError("Exhaustive layer digest drifted.")
        for layer in atlas["layers"]:
            count = layer["candidate_count"]
            if (
                count != len(layer["candidates"])
                or layer["core_cardinality"] != 0
                or layer["candidate_cardinality"] != 1
                or layer["full_cardinality"] != count
                or layer["deletion_cardinality"] != count - 1
                or sum(bool(item["is_task_evidence"]) for item in layer["candidates"]) != 1
            ):
                raise ValueError("Exhaustive counterfactual cardinality drifted.")
    elif atlas is not None:
        raise ValueError("Non-exhaustive cell unexpectedly contains an atlas.")


def _load_cells(
    manifest: dict[str, Any], manifest_path: Path, output_root: Path
) -> list[dict[str, Any]]:
    manifest_sha256 = runner._file_digest(manifest_path)
    expected = runner.coordinates(manifest)
    expected_paths = {runner.cell_path(output_root, coordinate) for coordinate in expected}
    observed_paths = set(output_root.rglob("replicate-*.json"))
    if observed_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - observed_paths)
        unexpected = sorted(str(path) for path in observed_paths - expected_paths)
        raise ValueError(
            f"Atlas output is not closed world: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    cells: list[dict[str, Any]] = []
    for coordinate in expected:
        cell = json.loads(runner.cell_path(output_root, coordinate).read_text())
        _validate_cell(cell, coordinate=coordinate, manifest_sha256=manifest_sha256)
        cells.append(cell)
    by_workload: dict[str, set[str]] = defaultdict(set)
    for cell in cells:
        by_workload[runner._digest(cell["workload_identity"])].add(cell["input_ids_sha256"])
    if any(len(digests) != 1 for digests in by_workload.values()):
        raise ValueError("Shared workload inputs differ across checkpoint seeds or scales.")
    return cells


def _select(
    cells: list[dict[str, Any]],
    *,
    scale: str | None = None,
    context: int | None = None,
    seed: int | None = None,
    family: str | None = None,
) -> list[dict[str, Any]]:
    return [
        cell
        for cell in cells
        if (scale is None or cell["coordinate"]["scale"] == scale)
        and (context is None or cell["coordinate"]["context"] == context)
        and (seed is None or cell["coordinate"]["training_seed"] == seed)
        and (family is None or cell["coordinate"]["family"] == family)
    ]


def _mean_delta(cells: list[dict[str, Any]], arm: str, metric: str) -> float:
    return _mean([_delta(cell, arm, metric) for cell in cells])


def _seed_deltas(
    cells: list[dict[str, Any]],
    *,
    scale: str,
    context: int,
    arm: str,
    metric: str,
) -> dict[int, float]:
    return {
        seed: _mean_delta(
            _select(cells, scale=scale, context=context, seed=seed), arm, metric
        )
        for seed in _MANIFEST["training_seeds"]
    }


def _topology(atlas_cells: list[dict[str, Any]]) -> dict[str, Any]:
    strata: dict[tuple[int, str, int, int], dict[int, int]] = defaultdict(dict)
    for cell in atlas_cells:
        coordinate = cell["coordinate"]
        for layer in cell["exhaustive_atlas"]["layers"]:
            top = max(
                layer["candidates"],
                key=lambda item: (
                    float(item["equal_budget_gold_log_prob_delta"]),
                    -int(item["block_end"]),
                ),
            )
            key = (
                coordinate["context"],
                coordinate["family"],
                coordinate["replicate"],
                layer["layer_index"],
            )
            strata[key][coordinate["training_seed"]] = int(top["block_end"])
    pairwise: list[float] = []
    for values in strata.values():
        if set(values) != set(_MANIFEST["training_seeds"]):
            raise ValueError("Causal topology stratum lacks five checkpoint seeds.")
        pairwise.extend(
            float(values[left] == values[right])
            for left, right in combinations(_MANIFEST["training_seeds"], 2)
        )
    return {
        "strata": len(strata),
        "pairwise_comparisons": len(pairwise),
        "mean_pairwise_singleton_jaccard": _mean(pairwise),
    }


def _observable_alignment(atlas_cells: list[dict[str, Any]]) -> dict[str, Any]:
    correlations: dict[int, list[float]] = defaultdict(list)
    teacher_correlations: dict[int, list[float]] = defaultdict(list)
    evidence_top_utility: list[float] = []
    evidence_top_necessity: list[float] = []
    for cell in atlas_cells:
        seed = cell["coordinate"]["training_seed"]
        for layer in cell["exhaustive_atlas"]["layers"]:
            candidates = layer["candidates"]
            proxy = [float(item["scores"]["perturbation-proxy"]) for item in candidates]
            utility = [
                float(item["equal_budget_gold_log_prob_delta"]) for item in candidates
            ]
            teacher_utility = [
                float(item["equal_budget_teacher_log_prob_delta"]) for item in candidates
            ]
            correlations[seed].append(spearman(proxy, utility))
            teacher_correlations[seed].append(spearman(proxy, teacher_utility))
            top_utility = max(
                candidates,
                key=lambda item: (
                    float(item["equal_budget_gold_log_prob_delta"]),
                    -int(item["block_end"]),
                ),
            )
            top_necessity = max(
                candidates,
                key=lambda item: (
                    float(item["necessity_gold_log_prob_loss"]),
                    -int(item["block_end"]),
                ),
            )
            evidence_top_utility.append(float(bool(top_utility["is_task_evidence"])))
            evidence_top_necessity.append(float(bool(top_necessity["is_task_evidence"])))
    seed_means = {seed: _mean(values) for seed, values in sorted(correlations.items())}
    teacher_seed_means = {
        seed: _mean(values) for seed, values in sorted(teacher_correlations.items())
    }
    return {
        "gold_seed_mean_spearman": seed_means,
        "teacher_seed_mean_spearman": teacher_seed_means,
        "strata_per_seed": {seed: len(values) for seed, values in correlations.items()},
        "evidence_is_top_gold_utility_rate": _mean(evidence_top_utility),
        "evidence_is_top_gold_necessity_rate": _mean(evidence_top_necessity),
    }


def analyze(manifest: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    draws = int(manifest["analysis_plan"]["bootstrap_draws"])
    bootstrap_seed = int(manifest["analysis_plan"]["bootstrap_seed"])
    seeds = list(manifest["training_seeds"])
    atlas_cells = [cell for cell in cells if cell["exhaustive_atlas"] is not None]
    topology = _topology(atlas_cells)
    alignment = _observable_alignment(atlas_cells)

    proxy_640 = _seed_deltas(
        cells,
        scale="s151",
        context=640,
        arm="perturbation-proxy",
        metric="target_log_prob",
    )
    oracle_640 = _seed_deltas(
        cells,
        scale="s151",
        context=640,
        arm="force-evidence",
        metric="target_log_prob",
    )
    recovered_fraction = _mean(list(proxy_640.values())) / _mean(list(oracle_640.values()))
    recovered_ci = _bootstrap_ratio(
        [proxy_640[seed] for seed in seeds],
        [oracle_640[seed] for seed in seeds],
        draws=draws,
        seed=bootstrap_seed,
    )

    proxy_1024 = _seed_deltas(
        cells,
        scale="s151",
        context=1024,
        arm="perturbation-proxy",
        metric="target_log_prob",
    )
    family_1024 = {
        family: _mean_delta(
            _select(cells, scale="s151", context=1024, family=family),
            "perturbation-proxy",
            "target_log_prob",
        )
        for family in runner.FAMILIES
    }

    s55_target = _seed_deltas(
        cells,
        scale="s55",
        context=640,
        arm="perturbation-proxy",
        metric="target_log_prob",
    )
    s55_accuracy = _seed_deltas(
        cells,
        scale="s55",
        context=640,
        arm="perturbation-proxy",
        metric="correct",
    )
    s55_target_ci = _bootstrap_mean(
        [s55_target[seed] for seed in seeds], draws=draws, seed=bootstrap_seed + 1
    )
    s55_accuracy_ci = _bootstrap_mean(
        [100.0 * s55_accuracy[seed] for seed in seeds],
        draws=draws,
        seed=bootstrap_seed + 2,
    )

    gate_topology = topology["mean_pairwise_singleton_jaccard"] >= 0.60
    gate_alignment = all(
        float(value) >= 0.30 for value in alignment["gold_seed_mean_spearman"].values()
    )
    gate_s151_640 = (
        all(value > 0.0 for value in proxy_640.values())
        and _mean(list(oracle_640.values())) > 0.0
        and recovered_fraction >= 0.25
        and recovered_ci[0] >= 0.10
    )
    gate_length = (
        sum(value > 0.0 for value in proxy_1024.values()) >= 4
        and sum(value > 0.0 for value in family_1024.values()) >= 3
    )
    gate_safety = s55_target_ci[0] > -0.10 and s55_accuracy_ci[0] > -2.0
    gates = {
        "causal_topology": gate_topology,
        "observable_alignment": gate_alignment,
        "s151_640_recovery": gate_s151_640,
        "s151_1024_length_shift": gate_length,
        "s55_2x_safety": gate_safety,
    }
    all_pass = all(gates.values())
    if all_pass:
        decision = "GO_NATURAL_LANGUAGE_TRANSFER_WITHOUT_SYNTHETIC_ROUTER_TRAINING"
    elif gate_topology and not (gate_alignment and gate_s151_640):
        decision = "OBSERVABLE_PROXY_FAILED_CONSIDER_SEPARATELY_APPROVED_NESTED_CV_RERANKER"
    elif not gate_topology:
        decision = "UNSTABLE_CAUSAL_TOPOLOGY_DO_NOT_TRAIN_SHARED_SELECTOR"
    else:
        decision = "NO_GO_RESTORE_OR_RECONSTRUCT_MEMORY"

    diagnostic_arms: dict[str, dict[str, float]] = {}
    for arm in runner.OBSERVABLE_POLICIES:
        diagnostic_arms[arm] = {
            "s151_640_target_log_prob_delta": _mean_delta(
                _select(cells, scale="s151", context=640), arm, "target_log_prob"
            ),
            "s151_1024_target_log_prob_delta": _mean_delta(
                _select(cells, scale="s151", context=1024), arm, "target_log_prob"
            ),
            "s55_640_2x_target_log_prob_delta": _mean_delta(
                _select(cells, scale="s55", context=640), arm, "target_log_prob"
            ),
        }
    return {
        "schema_version": 1,
        "experiment_id": runner.EXPERIMENT_ID,
        "cells": len(cells),
        "exhaustive_cells": len(atlas_cells),
        "integrity": {
            "closed_world": True,
            "identity_outcomes_equal": len(cells),
            "shared_workload_inputs_equal": True,
        },
        "causal_topology": topology,
        "observable_alignment": alignment,
        "primary": {
            "s151_640_proxy_minus_native_by_seed": proxy_640,
            "s151_640_oracle_minus_native_by_seed": oracle_640,
            "s151_640_recovered_oracle_fraction": recovered_fraction,
            "s151_640_recovered_fraction_bootstrap_95": list(recovered_ci),
            "s151_1024_proxy_minus_native_by_seed": proxy_1024,
            "s151_1024_proxy_minus_native_by_family": family_1024,
            "s55_2x_proxy_minus_native_target_log_prob_by_seed": s55_target,
            "s55_2x_target_log_prob_bootstrap_95": list(s55_target_ci),
            "s55_2x_proxy_minus_native_accuracy_pp_by_seed": {
                seed: 100.0 * value for seed, value in s55_accuracy.items()
            },
            "s55_2x_accuracy_pp_bootstrap_95": list(s55_accuracy_ci),
        },
        "diagnostic_arms": diagnostic_arms,
        "gates": gates,
        "all_primary_gates_pass": all_pass,
        "decision": decision,
        "bootstrap": {
            "independent_unit": "training checkpoint seed",
            "draws": draws,
            "seed": bootstrap_seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the causal-identifiability atlas.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    global _MANIFEST
    _MANIFEST = runner._load_manifest(args.manifest)
    if args.output_root.resolve() != Path(_MANIFEST["output_root"]).resolve():
        raise ValueError("Summary output root differs from the frozen manifest.")
    cells = _load_cells(_MANIFEST, args.manifest, args.output_root)
    result = analyze(_MANIFEST, cells)
    result["manifest_sha256"] = runner._file_digest(args.manifest)
    runner._atomic_json(args.summary, result)
    print(json.dumps({"status": "complete", "decision": result["decision"]}), flush=True)


_MANIFEST: dict[str, Any] = {}


if __name__ == "__main__":
    main()
