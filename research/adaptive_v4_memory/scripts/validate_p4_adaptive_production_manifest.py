from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

SCALES = ("s55", "s151")
BUDGETS = ("2x", "4x")
CONTEXTS = (8_192, 32_768, 131_072)
GENERATIONS = (128, 512, 2_048)
BATCHES = (1, 4, 8, 16)
CONCURRENCY = (1, 8, 32)
POLICIES = ("fixed+pins", "calibrated+pins")
WARMUPS = 5
REPETITIONS = 30
EXPECTED_CELLS = 432
INPUT_SEED_BASE = 9_271_400


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(payload: dict[str, Any]) -> None:
    _require(payload.get("schema_version") == 1, "Schema version drifted.")
    _require(
        payload.get("experiment_id") == "p4-adaptive-production-systems-matrix-v1",
        "Experiment identity drifted.",
    )
    _require(
        payload.get("status") == "frozen_before_any_adaptive_production_cell",
        "Adaptive production cohort is not frozen before execution.",
    )
    _require(tuple(payload.get("scales", ())) == SCALES, "Scale set drifted.")
    _require(tuple(payload.get("budgets", ())) == BUDGETS, "Budget set drifted.")
    _require(tuple(payload.get("paired_policies", ())) == POLICIES, "Policy pair drifted.")
    _require(tuple(payload.get("contexts_tokens", ())) == CONTEXTS, "Context set drifted.")
    _require(
        tuple(payload.get("generation_tokens", ())) == GENERATIONS,
        "Generation set drifted.",
    )
    profiles = payload.get("load_profiles", [])
    observed = {
        (row.get("batch"), row.get("concurrency"))
        for row in profiles
        if isinstance(row, dict)
    }
    _require(
        len(profiles) == 12
        and observed == set(product(BATCHES, CONCURRENCY))
        and len({row.get("name") for row in profiles}) == 12,
        "Batch-by-concurrency factorial drifted.",
    )
    expected_cells = (
        len(SCALES)
        * len(BUDGETS)
        * len(CONTEXTS)
        * len(GENERATIONS)
        * len(profiles)
    )
    _require(expected_cells == EXPECTED_CELLS, "Internal cell-count constant drifted.")
    _require(payload.get("primary_paired_cells") == expected_cells, "Cell count drifted.")
    _require(
        payload.get("warmups_per_paired_cell") == WARMUPS
        and payload.get("measured_repetitions_per_paired_cell") == REPETITIONS,
        "Warmup or repetition count drifted.",
    )
    _require(
        payload.get("primary_measured_policy_runs")
        == EXPECTED_CELLS * REPETITIONS * len(POLICIES)
        and payload.get("primary_total_policy_runs_including_warmup")
        == EXPECTED_CELLS * (WARMUPS + REPETITIONS) * len(POLICIES),
        "Policy-run volume drifted.",
    )
    _require(payload.get("input_seed_base") == INPUT_SEED_BASE, "Input seed drifted.")
    controller = payload.get("controller_contract", {})
    _require(
        isinstance(controller, dict)
        and "exact P2 causal factorial" in controller.get("source", "")
        and "batch x concurrency" in controller.get("batching", "")
        and "same global physical hot-block total" in controller.get("physical_budget", ""),
        "Exact controller or physical-budget contract drifted.",
    )
    statistics = payload.get("statistical_analysis", {})
    _require(
        statistics.get("paired_bootstrap_resamples") == 10_000
        and statistics.get("bootstrap_seed") == 9_271_502
        and "Holm" in statistics.get("multiplicity", "")
        and "never silently retried or dropped" in statistics.get("failure_policy", ""),
        "Statistical or failure protocol drifted.",
    )
    claim = payload.get("claim_boundary", "")
    _require(
        all(
            phrase in claim
            for phrase in (
                "simultaneous real GPU batches",
                "does not establish dynamic arrivals",
                "fused kernels",
                "kernel-aware residency layout",
                "position-aware recomputation costs",
                "official DeepSeek-V4",
                "Quality claims remain bound to P2 and P3",
            )
        ),
        "Adaptive production claim boundary drifted.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen adaptive P4 production cohort.")
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=Path(
            "research/adaptive_v4_memory/manifests/"
            "p4-adaptive-production-systems-matrix-v1.json"
        ),
    )
    args = parser.parse_args()
    validate_manifest(json.loads(args.manifest.read_text()))
    print(f"validated {EXPECTED_CELLS} adaptive production cells")


if __name__ == "__main__":
    main()
