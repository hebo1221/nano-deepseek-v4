from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import run_restoration_ceiling_e2a as runner


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


def _arm_delta(
    cell: dict[str, Any], arm: str, metric: str, *, baseline: str = "native"
) -> float:
    return _outcome_metric(cell, arm, metric) - _outcome_metric(cell, baseline, metric)


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
    sampled_bottom = bottom[indices].mean(axis=1)
    estimates = np.divide(
        top[indices].mean(axis=1),
        sampled_bottom,
        out=np.full(draws, -np.inf),
        where=sampled_bottom > 0.0,
    )
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def _validate_cell(
    cell: dict[str, Any], *, coordinate: dict[str, Any], manifest_sha256: str
) -> None:
    if (
        cell.get("schema_version") != 1
        or cell.get("experiment_id") != runner.EXPERIMENT_ID
        or cell.get("manifest_sha256") != manifest_sha256
        or cell.get("coordinate") != coordinate
    ):
        raise ValueError("Restoration-ceiling cell binding drifted.")
    if cell.get("workload_identity") != runner.workload_identity(coordinate):
        raise ValueError("Restoration-ceiling workload identity drifted.")
    if cell.get("evaluation_seed") != runner.evaluation_seed(
        coordinate, _MANIFEST["evaluation_seed_namespace"]
    ):
        raise ValueError("Restoration-ceiling evaluation seed drifted.")
    if set(cell.get("outcomes", {})) != set(runner.ARMS):
        raise ValueError("Restoration-ceiling arm coverage drifted.")
    route = cell["route_contract"]
    if not (
        route.get("identity_replay_bitwise_equal") is True
        and route.get("native_equivalent_replay_bitwise_equal") is True
        and route.get("observable_routes_exclude_evidence_and_targets") is True
    ):
        raise ValueError("Restoration-ceiling replay invariant is absent.")
    if cell["outcomes"]["identity-replay"] != cell["outcomes"]["native"]:
        raise ValueError("Stored identity outcome differs from native.")
    equivalent = route["native_equivalent_arm"]
    if cell["outcomes"][equivalent] != cell["outcomes"]["native"]:
        raise ValueError("Stored native-equivalent absolute-K outcome differs from native.")
    if runner._digest(route["observable_receipts"]) != route["observable_receipts_sha256"]:
        raise ValueError("Observable receipt digest drifted.")
    if runner._digest(route["oracle_receipts"]) != route["oracle_receipts_sha256"]:
        raise ValueError("Evidence receipt digest drifted.")
    native_topk = int(route["native_topk"])
    for receipt in route["observable_receipts"]:
        selected = receipt["selected_block_ends"]
        if set(selected) != {"native", *runner.OBSERVABLE_ARMS}:
            raise ValueError("Observable route receipt coverage drifted.")
        if len(selected["native"]) != native_topk:
            raise ValueError("Native receipt cardinality drifted.")
        for topk in runner.ABSOLUTE_INDEX_K:
            ends = selected[f"index-k{topk}"]
            if len(ends) != topk or len(set(ends)) != topk:
                raise ValueError("Absolute-K route cardinality drifted.")
        full = selected["full-compressed"]
        if len(full) != receipt["eligible_blocks"] or len(set(full)) != len(full):
            raise ValueError("Full-compressed route coverage drifted.")
    for receipt in route["oracle_receipts"]:
        if len(receipt["selected_block_ends"]) != native_topk:
            raise ValueError("Evidence route cardinality drifted.")


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
            "Restoration-ceiling output is not closed world: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
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
        raise ValueError("Shared workloads differ across checkpoint seeds or scales.")
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


def _mean_delta(
    cells: list[dict[str, Any]], arm: str, metric: str, *, baseline: str = "native"
) -> float:
    return _mean([_arm_delta(cell, arm, metric, baseline=baseline) for cell in cells])


def _seed_deltas(
    cells: list[dict[str, Any]],
    *,
    scale: str,
    arm: str,
    metric: str,
    context: int | None = None,
    baseline: str = "native",
) -> dict[int, float]:
    return {
        seed: _mean_delta(
            _select(cells, scale=scale, context=context, seed=seed),
            arm,
            metric,
            baseline=baseline,
        )
        for seed in _MANIFEST["training_seeds"]
    }


def _decision(gates: dict[str, bool]) -> str:
    if all(gates.values()):
        return "GO_SEPARATELY_FROZEN_COMPACT_RESTORATION_EXPERIMENT"
    teacher_value_pass = gates["s151_640_full_teacher"] and gates[
        "s151_1024_length_shift"
    ]
    if teacher_value_pass and not gates["distributed_beyond_k8"]:
        return "PREFER_SMALL_FIXED_K_CAPACITY_STUDY_OVER_RESTORATION"
    if teacher_value_pass:
        return "CONDITIONAL_RESTORATION_REQUIRES_FALLBACK_GATE"
    return "NO_GO_FULL_CACHE_TEACHER_PIVOT_EXTERNAL_RETRIEVAL"


def analyze(manifest: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    draws = int(manifest["analysis_plan"]["bootstrap_draws"])
    bootstrap_seed = int(manifest["analysis_plan"]["bootstrap_seed"])
    seeds = list(manifest["training_seeds"])

    full_640 = _seed_deltas(
        cells,
        scale="s151",
        context=640,
        arm="full-compressed",
        metric="target_log_prob",
    )
    full_640_ci = _bootstrap_mean(
        [full_640[seed] for seed in seeds], draws=draws, seed=bootstrap_seed
    )
    full_1024 = _seed_deltas(
        cells,
        scale="s151",
        context=1024,
        arm="full-compressed",
        metric="target_log_prob",
    )
    full_1024_ci = _bootstrap_mean(
        [full_1024[seed] for seed in seeds], draws=draws, seed=bootstrap_seed + 1
    )
    full_1024_family = {
        family: _mean_delta(
            _select(cells, scale="s151", context=1024, family=family),
            "full-compressed",
            "target_log_prob",
        )
        for family in runner.FAMILIES
    }

    beyond_k8 = _seed_deltas(
        cells,
        scale="s151",
        arm="full-compressed",
        metric="target_log_prob",
        baseline="index-k8",
    )
    beyond_k8_ci = _bootstrap_mean(
        [beyond_k8[seed] for seed in seeds], draws=draws, seed=bootstrap_seed + 2
    )

    evidence_640 = _seed_deltas(
        cells,
        scale="s151",
        context=640,
        arm="force-evidence",
        metric="target_log_prob",
    )
    evidence_mean = _mean(list(evidence_640.values()))
    recovery = _mean(list(full_640.values())) / evidence_mean if evidence_mean > 0.0 else -math.inf
    recovery_ci = _bootstrap_ratio(
        [full_640[seed] for seed in seeds],
        [evidence_640[seed] for seed in seeds],
        draws=draws,
        seed=bootstrap_seed + 3,
    )

    s55_target = _seed_deltas(
        cells,
        scale="s55",
        context=640,
        arm="full-compressed",
        metric="target_log_prob",
    )
    s55_accuracy = _seed_deltas(
        cells,
        scale="s55",
        context=640,
        arm="full-compressed",
        metric="correct",
    )
    s55_target_ci = _bootstrap_mean(
        [s55_target[seed] for seed in seeds], draws=draws, seed=bootstrap_seed + 4
    )
    s55_accuracy_ci = _bootstrap_mean(
        [100.0 * s55_accuracy[seed] for seed in seeds],
        draws=draws,
        seed=bootstrap_seed + 5,
    )

    gates = {
        "s151_640_full_teacher": all(value > 0.0 for value in full_640.values())
        and full_640_ci[0] > 0.0,
        "s151_1024_length_shift": sum(value > 0.0 for value in full_1024.values()) >= 4
        and sum(value > 0.0 for value in full_1024_family.values()) >= 3
        and full_1024_ci[0] > 0.0,
        "distributed_beyond_k8": sum(value > 0.0 for value in beyond_k8.values()) >= 4
        and beyond_k8_ci[0] > 0.0,
        "evidence_headroom_recovery": evidence_mean > 0.0
        and recovery >= 0.50
        and recovery_ci[0] >= 0.25,
        "s55_2x_safety": s55_target_ci[0] > -0.10 and s55_accuracy_ci[0] > -2.0,
    }
    all_pass = all(gates.values())
    decision = _decision(gates)

    hard_curve: dict[str, dict[str, float]] = {}
    for arm in ("index-k1", "index-k2", "index-k4", "index-k8", "full-compressed"):
        hard_curve[arm] = {
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
        "integrity": {
            "closed_world": True,
            "identity_outcomes_equal": len(cells),
            "native_equivalent_outcomes_equal": len(cells),
            "shared_workload_inputs_equal": True,
        },
        "primary": {
            "s151_640_full_minus_native_by_seed": full_640,
            "s151_640_bootstrap_95": list(full_640_ci),
            "s151_1024_full_minus_native_by_seed": full_1024,
            "s151_1024_full_minus_native_by_family": full_1024_family,
            "s151_1024_bootstrap_95": list(full_1024_ci),
            "s151_full_minus_k8_by_seed": beyond_k8,
            "s151_full_minus_k8_bootstrap_95": list(beyond_k8_ci),
            "s151_640_evidence_minus_native_by_seed": evidence_640,
            "s151_640_full_recovered_evidence_fraction": recovery,
            "s151_640_recovery_bootstrap_95": list(recovery_ci),
            "s55_2x_full_minus_native_target_log_prob_by_seed": s55_target,
            "s55_2x_target_log_prob_bootstrap_95": list(s55_target_ci),
            "s55_2x_full_minus_native_accuracy_pp_by_seed": {
                seed: 100.0 * value for seed, value in s55_accuracy.items()
            },
            "s55_2x_accuracy_pp_bootstrap_95": list(s55_accuracy_ci),
        },
        "diagnostic_hard_capacity_curve": hard_curve,
        "diagnostic_long_generation_1024_full_minus_native": _mean_delta(
            _select(
                cells,
                scale="s151",
                context=1024,
                family="long-generation-changing-evidence",
            ),
            "full-compressed",
            "target_log_prob",
        ),
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
    parser = argparse.ArgumentParser(description="Summarize E2-A restoration ceiling.")
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
