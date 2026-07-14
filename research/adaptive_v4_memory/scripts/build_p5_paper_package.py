from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

ALLOWED_CLASSES = {"success", "bounded-result", "negative-result", "unverified"}
P4_EXPECTED_CELLS = 216
REQUIRED_TRACEABILITY_IDS = {
    *(f"P0.{index}" for index in range(1, 4)),
    *(f"P1.{index}" for index in range(1, 6)),
    *(f"P2.{index}" for index in range(1, 6)),
    *(f"P3.{index}" for index in range(1, 6)),
    *(f"P4.{index}" for index in range(1, 5)),
    *(f"P5.{index}" for index in range(1, 5)),
    *(f"C.{index}" for index in range(1, 7)),
}
TRACEABILITY_SOURCE_KINDS = {
    "evidence",
    "execution-audit",
    "boundary-manifest",
    "generated-output",
    "generator",
    "verification-contract",
}
FINAL_RELEASE_COMMANDS = [
    ".venv/bin/ruff check nano_deepseek_v4 research/adaptive_v4_memory/scripts tests",
    ".venv/bin/mypy nano_deepseek_v4 research/adaptive_v4_memory/scripts",
    ".venv/bin/pytest -q",
    ".venv/bin/python -m build",
    ".venv/bin/twine check dist/*",
]
P5_GENERATOR_PATH = Path(
    "research/adaptive_v4_memory/scripts/build_p5_paper_package.py"
)
CONTROLLER_CONTRACT_TESTS = {
    "deterministic_replay": (
        "tests/test_causal_memory_controller.py::"
        "test_same_token_controller_is_causal_bounded_and_deterministic"
    ),
    "cache_lifecycle_and_physical_tier_budget": (
        "tests/test_causal_memory_controller.py::"
        "test_same_token_cache_persistence_and_lifecycle"
    ),
    "dense_recovery": (
        "tests/test_causal_memory_controller.py::"
        "test_same_token_controller_dense_fallback_and_incomplete_group_guard"
    ),
}
REPRODUCTION_REQUIRED_MARKERS = [
    "run_p2_core_parallel.py --scale s55 --workers 3",
    "run_p2_core_parallel.py --scale s151 --workers 3",
    "summarize_p2_core_matrix.py",
    "run_p2_seed_extension_prerequisites.py",
    "run_p2_seed_extension_core.py --scale s55 --workers 3",
    "run_p2_seed_extension_core.py --scale s151 --workers 3",
    "summarize_p2_seed_extension.py",
    "run_p2_causal_prerequisites.py",
    "run_p2_causal_parallel.py --workers 3",
    "summarize_p2_causal_factorial.py",
    "run_p2_seed_extension_causal_prerequisites.py",
    "run_p2_seed_extension_causal.py --scale s55 --workers 3",
    "run_p2_seed_extension_causal.py --scale s151 --workers 3",
    "summarize_p2_seed_extension_causal.py",
    "run_p3_natural_ruler.py",
    "run_p3_natural_ruler.py --cohort adaptive-quota",
    "summarize_p3_natural_adaptive_quota_ruler.py",
    "validate_p3_natural_adaptive_quota_scbench_manifest.py",
    "run_p3_scbench.py --cohort adaptive-quota",
    "summarize_p3_natural_adaptive_quota_scbench.py",
    "validate_p3_natural_adaptive_quota_longbench_v2_manifest.py",
    "run_p3_longbench_v2.py --cohort adaptive-quota",
    "summarize_p3_natural_adaptive_quota_longbench_v2.py",
    "validate_p3_natural_adaptive_quota_longmemeval_manifest.py",
    "run_p3_longmemeval.py --cohort adaptive-quota",
    "summarize_p3_natural_adaptive_quota_longmemeval.py",
    "validate_p3_natural_adaptive_quota_mrcr_manifest.py",
    "run_p3_mrcr.py --cohort adaptive-quota",
    "summarize_p3_natural_adaptive_quota_mrcr.py",
    "validate_p3_natural_adaptive_quota_suite_manifest.py",
    "summarize_p3_natural_adaptive_quota_suite.py",
    "prepare_p3_cross_family_ruler_dataset.py",
    "run_p3_cross_family_ruler.py",
    "summarize_p3_cross_family_ruler.py",
    "run_p3_cross_family_ruler.py --cohort adaptive-quota",
    "summarize_p3_cross_family_adaptive_quota_ruler.py",
    "validate_p3_cross_family_adaptive_quota_longbench_v2_manifest.py",
    "run_p3_longbench_v2.py --cohort cross-family-adaptive-quota",
    "summarize_p3_cross_family_adaptive_quota_longbench_v2.py",
    "run_p3_scbench.py",
    "run_p3_longbench_v2.py",
    "run_p3_longmemeval.py",
    "run_p3_mrcr.py",
    "run_p3_safety_stress.py",
    "run_p4_500k_context_preflight.py",
    "run_p4_systems_matrix.py",
    "run_p4_adaptive_systems_matrix.py",
    "summarize_p4_adaptive_systems_matrix.py",
    "run_p4_adaptive_production_systems_matrix.py",
    "summarize_p4_adaptive_production_systems_matrix.py",
    "run_p4_production_systems_matrix.py",
    "run_p1_online_lookahead_parallel.py --workers 3",
    "build_p5_paper_package.py",
    "run_p5_release_gate.py",
    *FINAL_RELEASE_COMMANDS,
    "GitHub Actions remains disabled by user request",
    "Official DeepSeek-V4 boundary",
]
BOUNDARY_EXPERIMENT_IDS = {
    "paper_grade_study": "adaptive-v4-memory-paper-grade-v1",
    "experiment_scale_audit": "adaptive-v4-memory-experiment-scale-audit-v1",
    "p2_seed_extension": "p2-independent-seed-extension-v1",
    "p2_causal_factorial": "p2-causal-factorial-v1",
    "online_learned_lookahead": "p1-online-learned-lookahead-v1",
    "p3_ruler": "p3-ruler-qwen3-1.7b-v1",
    "cross_family": "p3-cross-family-ruler-transfer-v1",
    "cross_family_adaptive_quota": "p3-cross-family-adaptive-quota-ruler-v1",
    "cross_family_adaptive_quota_longbench_v2": (
        "p3-cross-family-adaptive-quota-longbench-v2-v1"
    ),
    "natural_adaptive_quota": "p3-natural-adaptive-quota-ruler-v1",
    "natural_adaptive_quota_scbench": "p3-natural-adaptive-quota-scbench-v1",
    "natural_adaptive_quota_longbench_v2": (
        "p3-natural-adaptive-quota-longbench-v2-v1"
    ),
    "natural_adaptive_quota_longmemeval": (
        "p3-natural-adaptive-quota-longmemeval-v1"
    ),
    "natural_adaptive_quota_mrcr": "p3-natural-adaptive-quota-mrcr-v1",
    "natural_adaptive_quota_suite": "p3-natural-adaptive-quota-suite-v1",
    "natural_suite": "p3-natural-language-suite-v1",
    "safety_stress": "p3-qwen3-4b-safety-stress-v1",
    "natural_safety": "p3-qwen3-4b-natural-safety-v1",
    "p4_500k_context": "p4-500k-context-preflight-v1",
    "p4_reference_systems": "p4-reference-systems-matrix-v1",
    "p4_adaptive_systems": "p4-adaptive-systems-matrix-v1",
    "p4_adaptive_production_systems": "p4-adaptive-production-systems-matrix-v1",
    "p4_production_systems": "p4-production-systems-matrix-v1",
    "official_deepseek_v4": "p3-official-flashmemory-deepseek-v4-v1",
    "production_runtime_blocker": "p4-production-resource-blocker-v1",
}

SCALE_AUDIT_SOURCE_MANIFESTS = {
    "study": Path("research/adaptive_v4_memory/manifests/paper-grade-study-v1.json"),
    "causal": Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json"),
    "seed_extension": Path(
        "research/adaptive_v4_memory/manifests/p2-independent-seed-extension-v1.json"
    ),
    "online": Path("research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json"),
    "ruler": Path("research/adaptive_v4_memory/manifests/p3-ruler-qwen3-1.7b-v1.json"),
    "natural": Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    "cross_family": Path(
        "research/adaptive_v4_memory/manifests/p3-cross-family-ruler-transfer-v1.json"
    ),
    "cross_family_adaptive_quota": Path(
        "research/adaptive_v4_memory/manifests/"
        "p3-cross-family-adaptive-quota-ruler-v1.json"
    ),
    "cross_family_adaptive_quota_longbench_v2": Path(
        "research/adaptive_v4_memory/manifests/"
        "p3-cross-family-adaptive-quota-longbench-v2-v1.json"
    ),
    "natural_adaptive_quota": Path(
        "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-ruler-v1.json"
    ),
    "natural_adaptive_quota_scbench": Path(
        "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-scbench-v1.json"
    ),
    "natural_adaptive_quota_longbench_v2": Path(
        "research/adaptive_v4_memory/manifests/"
        "p3-natural-adaptive-quota-longbench-v2-v1.json"
    ),
    "natural_adaptive_quota_longmemeval": Path(
        "research/adaptive_v4_memory/manifests/"
        "p3-natural-adaptive-quota-longmemeval-v1.json"
    ),
    "natural_adaptive_quota_mrcr": Path(
        "research/adaptive_v4_memory/manifests/p3-natural-adaptive-quota-mrcr-v1.json"
    ),
    "safety": Path("research/adaptive_v4_memory/manifests/p3-safety-stress-v1.json"),
    "natural_safety": Path("research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"),
    "p4_reference": Path(
        "research/adaptive_v4_memory/manifests/p4-reference-systems-matrix-v1.json"
    ),
    "p4_adaptive": Path("research/adaptive_v4_memory/manifests/p4-adaptive-systems-matrix-v1.json"),
    "p4_preflight": Path("research/adaptive_v4_memory/manifests/p4-500k-context-preflight-v1.json"),
    "p4_production": Path(
        "research/adaptive_v4_memory/manifests/p4-production-systems-matrix-v1.json"
    ),
}
SCALE_AUDIT_CORE_DESIGN = Path("research/adaptive_v4_memory/scripts/evaluate_p2_core_shard.py")
P2_ANALYSIS_PATHS = {
    "p2_core": ("research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py",),
    "p2_core_confirmatory": (
        "research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py",
        "research/adaptive_v4_memory/scripts/summarize_p2_seed_extension.py",
    ),
    "p2_causal": (
        "research/adaptive_v4_memory/scripts/summarize_p2_causal_factorial.py",
        "research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py",
    ),
    "p2_causal_confirmatory": (
        "research/adaptive_v4_memory/scripts/summarize_p2_causal_factorial.py",
        "research/adaptive_v4_memory/scripts/summarize_p2_core_matrix.py",
        "research/adaptive_v4_memory/scripts/summarize_p2_seed_extension.py",
        "research/adaptive_v4_memory/scripts/summarize_p2_seed_extension_causal.py",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"Missing required P5 input: {path}")
    payload = json.loads(path.read_text())
    _require(isinstance(payload, dict), f"P5 input is not a JSON object: {path}")
    return payload


def _analysis_implementation_metadata(paths: tuple[str, ...]) -> dict[str, Any]:
    """Recompute the Git-index binding for code that produced P2 statistics."""

    canonical_paths = tuple(sorted(paths))
    tracked = subprocess.run(
        ["git", "ls-files", "-s", "--", *canonical_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    observed = tuple(line.split("\t", 1)[1] for line in tracked.splitlines() if "\t" in line)
    _require(
        observed == canonical_paths,
        f"P2 analysis implementation paths are untracked or reordered: {observed}",
    )
    return {
        "paths": list(canonical_paths),
        "tracked_file_count": len(canonical_paths),
        "git_index_sha256": hashlib.sha256(tracked.encode()).hexdigest(),
    }


def _paper_package_generator_input() -> dict[str, str]:
    """Bind generated tables and figures to the exact generator source."""

    path = P5_GENERATOR_PATH
    _require(
        path.is_file() and path.resolve() == Path(__file__).resolve(),
        f"P5 generator path drifted: {path}",
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(
        tracked.returncode == 0 and tracked.stdout.strip() == str(path),
        f"P5 generator is not tracked: {path}",
    )
    return {
        "name": "paper_package_generator",
        "kind": "generator",
        "path": str(path),
        "sha256": sha256(path),
    }


def _literal_assignment(path: Path, name: str) -> Any:
    """Load one top-level literal without importing an experiment runner."""

    _require(path.is_file(), f"Missing scale-audit design source: {path}")
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (TypeError, ValueError) as error:
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "tuple"
                    and len(value.args) == 1
                    and isinstance(value.args[0], ast.Call)
                    and isinstance(value.args[0].func, ast.Name)
                    and value.args[0].func.id == "range"
                    and not value.args[0].keywords
                    and all(
                        isinstance(argument, ast.Constant) and isinstance(argument.value, int)
                        for argument in value.args[0].args
                    )
                ):
                    range_args = [
                        cast(int, cast(ast.Constant, argument).value)
                        for argument in value.args[0].args
                    ]
                    return tuple(range(*range_args))
                raise ValueError(f"Scale-audit design field {name} is not literal.") from error
    raise ValueError(f"Scale-audit design field {name} is missing from {path}.")


def _validate_experiment_scale_audit(payload: dict[str, Any]) -> None:
    """Recompute every headline planned-volume count from frozen manifests."""

    sources = {name: _load(path) for name, path in SCALE_AUDIT_SOURCE_MANIFESTS.items()}
    planned = payload.get("planned_volume")
    _require(isinstance(planned, dict), "Experiment-scale planned volume is missing.")
    planned = cast(dict[str, Any], planned)

    study = sources["study"]
    causal = sources["causal"]
    training_seeds = len(study.get("training_seeds", []))
    scales = len(study.get("scales", []))
    families = len(study.get("workload_families", []))
    contexts = len(study.get("context_lengths", []))
    examples_per_family = study.get("minimum_examples_per_seed_scale_family")
    examples_per_shard = _literal_assignment(SCALE_AUDIT_CORE_DESIGN, "EXAMPLES_PER_SHARD")
    core_policies = _literal_assignment(SCALE_AUDIT_CORE_DESIGN, "CORE_POLICIES")
    frozen_replicates = _literal_assignment(SCALE_AUDIT_CORE_DESIGN, "REPLICATES")
    _require(
        isinstance(examples_per_family, int)
        and isinstance(examples_per_shard, int)
        and isinstance(core_policies, tuple)
        and isinstance(frozen_replicates, tuple)
        and contexts > 0
        and examples_per_family % (contexts * examples_per_shard) == 0,
        "P2 core shard derivation is not integral.",
    )
    examples_per_family = cast(int, examples_per_family)
    examples_per_shard = cast(int, examples_per_shard)
    core_policies = cast(tuple[str, ...], core_policies)
    frozen_replicates = cast(tuple[int, ...], frozen_replicates)
    replicates = examples_per_family // (contexts * examples_per_shard)
    _require(
        replicates == len(frozen_replicates),
        "P2 core replicate count drifted between protocol and runner.",
    )
    core_shards = training_seeds * scales * families * contexts * replicates
    expected_core = {
        "independent_shards": core_shards,
        "policy_example_evaluations": core_shards * examples_per_shard * len(core_policies),
        "training_seeds": training_seeds,
        "scales": scales,
        "workload_families": families,
        "context_lengths": contexts,
        "examples_per_seed_scale_family": examples_per_family,
    }
    _require(planned.get("p2_core") == expected_core, "P2 core scale count drifted.")

    primary_names = set(causal.get("primary_arms", {}))
    supplemental_names = set(causal.get("supplemental_baseline_arms", {}))
    contrast_names = {
        arm for pair in causal.get("component_contrasts", {}).values() for arm in pair
    }
    component_only_names = contrast_names - primary_names - supplemental_names
    causal_arm_count = len(primary_names | supplemental_names | component_only_names)
    budget_count = len(causal.get("primary_budget_points", []))
    causal_shards = core_shards * budget_count
    expected_causal = {
        "independent_shards": causal_shards,
        "policy_example_evaluations": causal_shards * examples_per_shard * causal_arm_count,
        "training_seeds": len(causal.get("training_seeds", [])),
        "scales": len(causal.get("scales", [])),
        "factorial_arms": causal_arm_count,
        "fixed_top_p_thresholds": [0.5, 0.8],
        "offline_registered_arm_oracle": True,
    }
    _require(planned.get("p2_causal") == expected_causal, "P2 causal scale count drifted.")

    extension_manifest = sources["seed_extension"]
    primary_cohort = extension_manifest.get("primary_cohort", {})
    extension_cohort = extension_manifest.get("extension_cohort", {})
    primary_seeds = set(primary_cohort.get("training_seeds", []))
    extension_seeds = set(extension_cohort.get("training_seeds", []))
    _require(
        extension_manifest.get("status") == "preregistered_before_primary_outcome_inspection"
        and extension_manifest.get("outcome_blinding", {}).get("primary_outcome_summary_inspected")
        is False
        and primary_seeds == set(study.get("training_seeds", []))
        and primary_seeds.isdisjoint(extension_seeds)
        and extension_cohort.get("outcome_dependent_early_stopping") is False
        and extension_cohort.get("separate_artifact_namespace_required") is True,
        "P2 independent-seed extension boundary drifted.",
    )
    extension_core_shards = (
        len(extension_seeds)
        * len(extension_cohort.get("scales", []))
        * extension_cohort.get("families", 0)
        * extension_cohort.get("contexts", 0)
        * extension_cohort.get("replicates_per_context", 0)
    )
    expected_extension_volume = {
        "training_runs": len(extension_seeds) * len(extension_cohort.get("scales", [])),
        "core_shards": extension_core_shards,
        "core_policy_example_evaluations": extension_core_shards
        * extension_cohort.get("examples_per_shard", 0)
        * len(core_policies),
        "causal_shards": extension_core_shards * budget_count,
        "causal_policy_example_evaluations": extension_core_shards
        * budget_count
        * extension_cohort.get("examples_per_shard", 0)
        * causal_arm_count,
    }
    _require(
        extension_manifest.get("planned_extension_volume") == expected_extension_volume
        and planned.get("p2_independent_seed_extension") == expected_extension_volume,
        "P2 independent-seed extension volume drifted.",
    )
    expected_combined_volume = {
        "independent_training_seeds_per_scale": len(primary_seeds | extension_seeds),
        "core_shards": core_shards + extension_core_shards,
        "core_policy_example_evaluations": expected_core["policy_example_evaluations"]
        + expected_extension_volume["core_policy_example_evaluations"],
        "causal_shards": causal_shards + expected_extension_volume["causal_shards"],
        "causal_policy_example_evaluations": expected_causal["policy_example_evaluations"]
        + expected_extension_volume["causal_policy_example_evaluations"],
    }
    _require(
        planned.get("combined_p2_confirmatory") == expected_combined_volume
        and extension_manifest.get("combined_p2_volume")
        == {
            key: value
            for key, value in expected_combined_volume.items()
            if key != "independent_training_seeds_per_scale"
        },
        "Combined P2 confirmatory volume drifted.",
    )

    online = sources["online"]
    online_matrix = online.get("matrix", {})
    online_seed_scales = len(online_matrix.get("scales", [])) * len(
        online.get("splits", {}).get("training_model_seeds", [])
    )
    training_examples = online_matrix.get("minimum_training_examples_per_seed_scale")
    calibration_examples = online_matrix.get("minimum_calibration_examples_per_seed_scale")
    _require(
        isinstance(training_examples, int)
        and isinstance(calibration_examples, int)
        and (training_examples + calibration_examples) % examples_per_shard == 0,
        "Online-lookahead label shard derivation is not integral.",
    )
    online_arms = len(online.get("arms", []))
    test_examples_per_arm = online_matrix.get("test_examples_per_arm")
    expected_online = {
        "label_shards": online_seed_scales
        * (training_examples + calibration_examples)
        // examples_per_shard,
        "fitted_policies": online_seed_scales * len(online_matrix.get("budgets", [])),
        "heldout_test_shards": online_seed_scales
        * len(online_matrix.get("workload_families", []))
        * len(online_matrix.get("contexts", []))
        * len(online_matrix.get("budgets", []))
        * online_matrix.get("replicates_per_context", 0),
        "paired_test_conversations_per_arm": test_examples_per_arm,
        "quality_arm_conversations": test_examples_per_arm * online_arms,
        "training_seeds": len(online.get("splits", {}).get("training_model_seeds", [])),
        "scales": len(online_matrix.get("scales", [])),
        "workload_families": len(online_matrix.get("workload_families", [])),
        "context_lengths": len(online_matrix.get("contexts", [])),
        "budgets": len(online_matrix.get("budgets", [])),
        "arms": online_arms,
    }
    _require(
        planned.get("p1_online_learned_lookahead") == expected_online,
        "Online-lookahead scale count drifted.",
    )

    ruler = sources["ruler"]
    ruler_benchmark = ruler.get("benchmark", {})
    ruler_cells_per_length = sum(
        len(arm.get("compression_ratios", [])) for arm in ruler.get("arms", [])
    )
    ruler_cells = ruler_cells_per_length * len(ruler_benchmark.get("lengths_tokens", []))
    expected_ruler = {
        "predictions": ruler_cells * ruler_benchmark.get("examples_per_length", 0),
        "model_scales": 1,
        "cells": ruler_cells,
    }
    _require(
        planned.get("p3_small_model_ruler") == expected_ruler,
        "P3 small-model RULER scale count drifted.",
    )

    natural = sources["natural"]
    natural_totals = natural.get("execution_totals", {})
    natural_protocol = natural.get("common_protocol", {})
    natural_benchmarks = natural.get("benchmarks", {})
    natural_arms = len(natural_protocol.get("mandatory_compatible_arms", []))
    expected_natural = {
        "predictions": natural_totals.get("minimum_predictions_per_arm", 0) * natural_arms,
        "benchmarks": len(natural_benchmarks),
        "required_arms": natural_arms,
        "model_families": 1,
        "maximum_primary_context_tokens": max(
            natural_benchmarks.get("RULER", {}).get("lengths_tokens", [])
        ),
    }
    _require(
        planned.get("p3_natural_qwen3_4b") == expected_natural,
        "P3 natural-suite scale count drifted.",
    )

    cross_family = sources["cross_family"]
    cross_benchmark = cross_family.get("benchmark", {})
    cross_relationship = cross_family.get("relationship_to_primary", {})
    expected_cross_family = {
        "predictions": cross_benchmark.get("paired_predictions_total"),
        "predictions_per_arm": cross_benchmark.get("predictions_per_arm"),
        "paired_arms": len(cross_family.get("arms", {})),
        "model_families_added": 1,
        "tasks": len(cross_benchmark.get("tasks", [])),
        "context_lengths": cross_benchmark.get("lengths_tokens", []),
        "examples_per_task_context_arm": cross_benchmark.get("samples_per_task_length"),
        "pooled_with_primary": cross_relationship.get("pooled_with_primary"),
        "phi_specific_tuning_allowed": cross_relationship.get("phi_specific_tuning_allowed"),
    }
    _require(
        planned.get("p3_cross_family_phi4_mini_ruler") == expected_cross_family,
        "P3 cross-family scale count drifted.",
    )

    cross_adaptive = sources["cross_family_adaptive_quota"]
    cross_adaptive_benchmark = cross_adaptive.get("benchmark", {})
    cross_adaptive_relationship = cross_adaptive.get("relationship_to_other_cohorts", {})
    expected_cross_adaptive = {
        "predictions": cross_adaptive_benchmark.get("paired_predictions_total"),
        "predictions_per_arm": cross_adaptive_benchmark.get("predictions_per_arm"),
        "paired_arms": len(cross_adaptive.get("arms", {})),
        "tasks": len(cross_adaptive_benchmark.get("tasks", [])),
        "context_lengths": cross_adaptive_benchmark.get("lengths_tokens", []),
        "examples_per_task_context_arm": cross_adaptive_benchmark.get(
            "samples_per_task_length"
        ),
        "same_global_token_budget": cross_adaptive.get("physical_contract", {}).get(
            "same_global_kept_tokens"
        ),
        "pooled_with_qwen": cross_adaptive_relationship.get("pooled_with_qwen"),
        "phi_specific_tuning_allowed": cross_adaptive_relationship.get(
            "phi_specific_tuning_or_reselection_allowed"
        ),
        "unchanged_synthetic_controller_transfer": False,
    }
    _require(
        planned.get("p3_cross_family_adaptive_quota_phi4_mini_ruler")
        == expected_cross_adaptive,
        "P3 cross-family adaptive-quota scale count drifted.",
    )

    cross_adaptive_longbench = sources["cross_family_adaptive_quota_longbench_v2"]
    cross_adaptive_longbench_benchmark = cross_adaptive_longbench.get("benchmark", {})
    cross_adaptive_longbench_lifecycle = cross_adaptive_longbench.get(
        "cache_lifecycle_contract", {}
    )
    cross_adaptive_longbench_relationship = cross_adaptive_longbench.get(
        "relationship_to_other_cohorts", {}
    )
    expected_cross_adaptive_longbench = {
        "predictions": cross_adaptive_longbench_benchmark.get(
            "paired_predictions_total"
        ),
        "predictions_per_arm": cross_adaptive_longbench_benchmark.get(
            "predictions_per_arm"
        ),
        "paired_arms": len(cross_adaptive_longbench.get("arms", {})),
        "categories": len(cross_adaptive_longbench_benchmark.get("categories", [])),
        "maximum_supported_context_tokens": cross_adaptive_longbench.get("model", {}).get(
            "maximum_supported_context_tokens"
        ),
        "same_initial_global_token_budget": cross_adaptive_longbench_lifecycle.get(
            "same_initial_global_kept_tokens"
        ),
        "pooled_with_qwen": cross_adaptive_longbench_relationship.get(
            "pooled_with_qwen_longbench_v2"
        ),
        "phi_specific_tuning_allowed": cross_adaptive_longbench_relationship.get(
            "phi_specific_tuning_or_reselection_allowed"
        ),
        "continuous_refresh_claim_available": cross_adaptive_longbench_lifecycle.get(
            "continuous_refresh_claim_available"
        ),
        "secondary_slices_are_descriptive": cross_adaptive_longbench.get(
            "statistics", {}
        ).get("secondary_slices_are_descriptive"),
    }
    _require(
        planned.get("p3_cross_family_adaptive_quota_phi4_mini_longbench_v2")
        == expected_cross_adaptive_longbench,
        "P3 cross-family adaptive-quota LongBench-v2 scale count drifted.",
    )

    natural_adaptive = sources["natural_adaptive_quota"]
    adaptive_benchmark = natural_adaptive.get("benchmark", {})
    expected_natural_adaptive = {
        "predictions": adaptive_benchmark.get("paired_predictions_total"),
        "predictions_per_arm": adaptive_benchmark.get("predictions_per_arm"),
        "paired_arms": len(natural_adaptive.get("arms", {})),
        "tasks": adaptive_benchmark.get("tasks"),
        "context_lengths": adaptive_benchmark.get("lengths_tokens"),
        "examples_per_task_context_arm": adaptive_benchmark.get("samples_per_task_length"),
        "same_global_token_budget": natural_adaptive.get("pins", {}).get("same_budget"),
        "unchanged_synthetic_controller_transfer": False,
    }
    _require(
        planned.get("p3_natural_adaptive_quota_qwen3_4b") == expected_natural_adaptive,
        "P3 natural adaptive-quota scale count drifted.",
    )

    adaptive_scbench = sources["natural_adaptive_quota_scbench"]
    adaptive_scbench_benchmark = adaptive_scbench.get("benchmark", {})
    adaptive_scbench_lifecycle = adaptive_scbench.get("cache_lifecycle_contract", {})
    expected_adaptive_scbench = {
        "predictions": adaptive_scbench_benchmark.get("paired_predictions_total"),
        "predictions_per_arm": adaptive_scbench_benchmark.get("predictions_per_arm"),
        "paired_arms": len(adaptive_scbench.get("arms", {})),
        "modes": len(adaptive_scbench_benchmark.get("modes", [])),
        "tasks": len(adaptive_scbench_benchmark.get("tasks", [])),
        "shared_context_rows_per_mode": adaptive_scbench_benchmark.get(
            "shared_context_rows_per_mode"
        ),
        "turn_predictions_per_mode": adaptive_scbench_benchmark.get(
            "turn_predictions_per_mode"
        ),
        "same_initial_global_token_budget": adaptive_scbench_lifecycle.get(
            "same_initial_global_kept_tokens"
        ),
        "continuous_refresh_claim_available": adaptive_scbench_lifecycle.get(
            "continuous_refresh_claim_available"
        ),
    }
    _require(
        planned.get("p3_natural_adaptive_quota_scbench_qwen3_4b")
        == expected_adaptive_scbench,
        "P3 adaptive SCBench scale count drifted.",
    )

    adaptive_longbench = sources["natural_adaptive_quota_longbench_v2"]
    adaptive_longbench_benchmark = adaptive_longbench.get("benchmark", {})
    adaptive_longbench_lifecycle = adaptive_longbench.get(
        "cache_lifecycle_contract", {}
    )
    expected_adaptive_longbench = {
        "predictions": adaptive_longbench_benchmark.get("paired_predictions_total"),
        "predictions_per_arm": adaptive_longbench_benchmark.get(
            "predictions_per_arm"
        ),
        "paired_arms": len(adaptive_longbench.get("arms", {})),
        "categories": len(adaptive_longbench_benchmark.get("categories", [])),
        "same_initial_global_token_budget": adaptive_longbench_lifecycle.get(
            "same_initial_global_kept_tokens"
        ),
        "continuous_refresh_claim_available": adaptive_longbench_lifecycle.get(
            "continuous_refresh_claim_available"
        ),
        "secondary_slices_are_descriptive": adaptive_longbench.get("statistics", {}).get(
            "secondary_slices_are_descriptive"
        ),
    }
    _require(
        planned.get("p3_natural_adaptive_quota_longbench_v2_qwen3_4b")
        == expected_adaptive_longbench,
        "P3 adaptive LongBench-v2 scale count drifted.",
    )

    adaptive_longmemeval = sources["natural_adaptive_quota_longmemeval"]
    adaptive_longmemeval_benchmark = adaptive_longmemeval.get("benchmark", {})
    adaptive_longmemeval_lifecycle = adaptive_longmemeval.get(
        "cache_lifecycle_contract", {}
    )
    expected_adaptive_longmemeval = {
        "predictions": adaptive_longmemeval_benchmark.get("paired_predictions_total"),
        "predictions_per_arm": adaptive_longmemeval_benchmark.get("examples_per_arm"),
        "paired_arms": len(adaptive_longmemeval.get("arms", {})),
        "same_initial_global_token_budget": adaptive_longmemeval_lifecycle.get(
            "same_initial_global_kept_tokens"
        ),
        "official_judge_available": adaptive_longmemeval.get(
            "official_metric_contract", {}
        ).get("initial_execution_mode")
        != "blocked",
        "quality_claim_available": adaptive_longmemeval.get(
            "official_metric_contract", {}
        ).get("confirmation_gate_available_before_official_judging"),
        "proxy_metric_substitution": False,
        "continuous_refresh_claim_available": adaptive_longmemeval_lifecycle.get(
            "continuous_refresh_claim_available"
        ),
    }
    _require(
        planned.get("p3_natural_adaptive_quota_longmemeval_qwen3_4b")
        == expected_adaptive_longmemeval,
        "P3 adaptive LongMemEval scale count drifted.",
    )

    adaptive_mrcr = sources["natural_adaptive_quota_mrcr"]
    adaptive_mrcr_benchmark = adaptive_mrcr.get("benchmark", {})
    adaptive_mrcr_lifecycle = adaptive_mrcr.get("cache_lifecycle_contract", {})
    needle_counts = len(adaptive_mrcr_benchmark.get("needle_counts", []))
    token_bins = adaptive_mrcr_benchmark.get("primary_token_bins")
    expected_adaptive_mrcr = {
        "predictions": adaptive_mrcr_benchmark.get("paired_predictions_total"),
        "predictions_per_arm": adaptive_mrcr_benchmark.get("predictions_per_arm"),
        "paired_arms": len(adaptive_mrcr.get("arms", {})),
        "needle_counts": needle_counts,
        "token_bins": token_bins,
        "needle_token_bin_cells": needle_counts * token_bins,
        "samples_per_cell_per_arm": adaptive_mrcr_benchmark.get(
            "samples_per_needle_bin_arm"
        ),
        "same_initial_global_token_budget": adaptive_mrcr_lifecycle.get(
            "same_initial_global_kept_tokens"
        ),
        "continuous_refresh_claim_available": adaptive_mrcr_lifecycle.get(
            "continuous_refresh_claim_available"
        ),
    }
    _require(
        planned.get("p3_natural_adaptive_quota_mrcr_qwen3_4b")
        == expected_adaptive_mrcr,
        "P3 adaptive MRCR scale count drifted.",
    )

    safety = sources["safety"]
    expected_safety = {
        "predictions": safety.get("expected_examples_per_arm", 0) * len(safety.get("arms", [])),
        "families": len(safety.get("families", [])),
        "context_lengths": len(safety.get("context_targets_tokens", [])),
        "paired_arms": len(safety.get("arms", [])),
        "examples_per_family_context_arm": safety.get("examples_per_family_context"),
    }
    _require(
        planned.get("p3_safety_qwen3_4b") == expected_safety,
        "P3 safety scale count drifted.",
    )

    natural_safety = sources["natural_safety"]
    safety_arms = len(natural_safety.get("required_arms", []))
    benchmarks = natural_safety.get("benchmarks", {})
    longsafety_per_arm = (
        benchmarks.get("LongSafety", {})
        .get("prompt_protocol", {})
        .get("expected_predictions_per_arm", 0)
    )
    ifeval_per_arm = (
        benchmarks.get("IFEval", {}).get("protocol", {}).get("expected_prompts_per_arm", 0)
    )
    expected_natural_safety = {
        "actual_model_generations": (longsafety_per_arm + ifeval_per_arm) * safety_arms,
        "longsafety_generations": longsafety_per_arm * safety_arms,
        "ifeval_generations_and_official_scores": ifeval_per_arm * safety_arms,
        "paired_arms": safety_arms,
        "longsafety_source_examples": sum(
            row.get("rows", 0)
            for row in benchmarks.get("LongSafety", {}).get("dataset", {}).get("files", [])
        ),
        "ifeval_prompts_per_arm": ifeval_per_arm,
        "official_longsafety_judge": "blocked_pending_explicit_paid_api_opt_in",
    }
    _require(
        planned.get("p3_natural_safety_qwen3_4b") == expected_natural_safety,
        "P3 natural-safety scale count drifted.",
    )

    for source_name, planned_name in (
        ("p4_reference", "p4_reference"),
        ("p4_production", "p4_production"),
    ):
        systems = sources[source_name]
        cells = (
            len(systems.get("scales", []))
            * len(systems.get("contexts_tokens", []))
            * len(systems.get("generation_tokens", []))
            * len(systems.get("load_profiles", []))
        )
        expected_systems = {
            "cells": cells,
            "warmups_per_cell": systems.get("warmups_per_paired_cell"),
            "measured_repetitions_per_cell": systems.get("measured_repetitions_per_paired_cell"),
            "policies": len(systems.get("policies", [])),
            "policy_runs_including_warmups": cells
            * (
                systems.get("warmups_per_paired_cell", 0)
                + systems.get("measured_repetitions_per_paired_cell", 0)
            )
            * len(systems.get("policies", [])),
        }
        _require(
            planned.get(planned_name) == expected_systems,
            f"{planned_name} scale count drifted.",
        )

    adaptive = sources["p4_adaptive"]
    adaptive_cells = (
        len(adaptive.get("scales", []))
        * len(adaptive.get("budgets", []))
        * len(adaptive.get("contexts_tokens", []))
        * len(adaptive.get("generation_tokens", []))
        * len(adaptive.get("load_profiles", []))
    )
    expected_adaptive = {
        "cells": adaptive_cells,
        "scales": len(adaptive.get("scales", [])),
        "budgets": len(adaptive.get("budgets", [])),
        "warmups_per_cell": adaptive.get("warmups_per_paired_cell"),
        "measured_repetitions_per_cell": adaptive.get("measured_repetitions_per_paired_cell"),
        "paired_policies": adaptive.get("paired_policies"),
        "policy_runs_including_warmups": adaptive_cells
        * (
            adaptive.get("warmups_per_paired_cell", 0)
            + adaptive.get("measured_repetitions_per_paired_cell", 0)
        )
        * len(adaptive.get("paired_policies", [])),
        "outcome_dependent_cell_selection": False,
    }
    _require(
        planned.get("p4_adaptive_reference") == expected_adaptive,
        "P4 adaptive reference scale count drifted.",
    )

    preflight = sources["p4_preflight"]
    expected_preflight = {
        "scale_cells": len(preflight.get("scales", [])),
        "policies": len(preflight.get("policies", [])),
        "terminal_policy_attempts": len(preflight.get("scales", []))
        * len(preflight.get("policies", []))
        * preflight.get("attempts_per_scale_policy", 0),
        "context_tokens": preflight.get("context_tokens"),
        "generation_tokens": preflight.get("generation_tokens"),
        "attempts_per_scale_policy": preflight.get("attempts_per_scale_policy"),
        "performance_claim_available": False,
    }
    _require(
        planned.get("p4_500k_context_preflight") == expected_preflight,
        "P4 500K preflight scale count drifted.",
    )
    reference = sources["p4_reference"]
    production = sources["p4_production"]
    study_systems = study.get("systems_matrix", {})
    reference_contexts = reference.get("contexts_tokens", [])
    production_contexts = production.get("contexts_tokens", [])
    registered_contexts = [*reference_contexts, preflight.get("context_tokens")]
    reference_batches = sorted({row.get("batch") for row in reference.get("load_profiles", [])})
    reference_loads = sorted(
        {row.get("active_requests") for row in reference.get("load_profiles", [])}
    )
    production_batches = sorted({row.get("batch") for row in production.get("load_profiles", [])})
    production_concurrency = sorted(
        {row.get("concurrency") for row in production.get("load_profiles", [])}
    )
    _require(
        study_systems.get("context_tokens") == registered_contexts
        and reference_contexts == production_contexts
        and study_systems.get("batch_sizes") == reference_batches == production_batches
        and study_systems.get("concurrency") == reference_loads == production_concurrency
        and study_systems.get("generation_tokens")
        == reference.get("generation_tokens")
        == production.get("generation_tokens"),
        "Paper-grade and executable P4 system grids drifted.",
    )
    independent_seeds = len(study.get("training_seeds", []))
    exact_assignments = 1 << independent_seeds
    expected_resolution = {
        "independent_training_seed_clusters_per_scale": independent_seeds,
        "exact_two_sided_sign_flip_assignments": exact_assignments,
        "minimum_attainable_two_sided_p": 2.0 / exact_assignments,
        "seed_cluster_bootstrap_resamples": study.get("statistics", {}).get("bootstrap_resamples"),
        "example_level_role": "paired descriptive precision within a training seed; examples do not increase the number of independent trained-model clusters",
        "p_value_used_as_success_gate": False,
        "interpretation": "The study has high within-seed sample density but only five independent training seeds per scale. Exact and multiplicity-adjusted seed-level p-values are reported, while causal success requires effect direction, corrected seed-cluster intervals, memory matching, and five-of-five seed consistency rather than an unattainable p<0.05 threshold.",
    }
    _require(
        payload.get("inference_resolution") == expected_resolution,
        "Experiment-scale independent-unit resolution drifted.",
    )
    combined_seed_count = len(primary_seeds | extension_seeds)
    combined_assignments = 1 << combined_seed_count
    expected_extension_resolution = {
        "independent_training_seed_clusters_per_scale": combined_seed_count,
        "exact_two_sided_sign_flip_assignments": combined_assignments,
        "minimum_attainable_two_sided_p": 2.0 / combined_assignments,
        "minimum_attainable_holm_adjusted_family_p": 9 * 2.0 / combined_assignments,
        "primary_cohort_remains_independently_reportable": True,
        "pooling_requires_identical_frozen_contracts": True,
        "outcome_dependent_early_stopping": False,
    }
    _require(
        payload.get("confirmatory_extension_resolution") == expected_extension_resolution,
        "Experiment-scale confirmatory seed resolution drifted.",
    )
    _require(
        "Do not add synthetic policy-example evaluations"
        in payload.get("non_aggregation_rule", ""),
        "Experiment-scale non-aggregation rule is missing.",
    )


def _validate_boundary_manifest(name: str, path: Path) -> dict[str, Any]:
    payload = _load(path)
    _require(name in BOUNDARY_EXPERIMENT_IDS, f"Unknown boundary manifest: {name}")
    _require(
        payload.get("experiment_id") == BOUNDARY_EXPERIMENT_IDS[name],
        f"Wrong {name} boundary experiment id.",
    )
    if name == "paper_grade_study":
        causal = payload.get("causal_ablation_matrix", {})
        hot_memory = causal.get("hot_memory_match", {})
        early_stop = payload.get("controller_early_stop", {})
        gate = payload.get("primary_causal_gate", {})
        extension = payload.get("confirmatory_seed_extension", {})
        _require(
            payload.get("protocol_version") == "2.1"
            and payload.get("scales") == ["s55", "s151"]
            and payload.get("training_seeds")
            == [6071401, 6071402, 6071403, 6071404, 6071405]
            and payload.get("evaluation_seeds")
            == [8071401, 8071402, 8071403, 8071404, 8071405]
            and payload.get("controller_calibration_seeds")
            == [7071401, 7071402, 7071403, 7071404, 7071405]
            and len(payload.get("workload_families", [])) == 9
            and payload.get("context_lengths") == [80, 128, 256, 512, 1024]
            and payload.get("minimum_examples_per_seed_scale_family") == 1_000,
            "Paper-grade primary matrix boundary drifted.",
        )
        _require(
            payload.get("required_ablations")
            == [
                "no-score-concentration",
                "no-temporal-reuse",
                "no-cross-layer-prior",
                "no-refresh-reuse",
                "no-protected-pins",
                "no-dense-fallback",
            ]
            and len(causal.get("arms", [])) == 10
            and hot_memory.get("maximum_relative_difference") == 0.01
            and hot_memory.get("retune_on") == "calibration-traces-only"
            and hot_memory.get("posthoc_accuracy_interpolation") is False,
            "Paper-grade causal-ablation boundary drifted.",
        )
        _require(
            early_stop.get("enabled") is False
            and early_stop.get("outcome_dependent") is False
            and early_stop.get("required_primary_seeds_per_scale") == 5
            and early_stop.get("required_extension_seeds_per_scale") == 4
            and early_stop.get("required_primary_causal_shards") == 9_000
            and early_stop.get("required_extension_causal_shards") == 7_200
            and gate.get("candidate") == "calibrated+pins"
            and gate.get("comparator") == "fixed+pins"
            and gate.get("same_measured_hot_memory_required") is True
            and gate.get("positive_seed_effects_required_per_scale") == 5
            and extension.get("combined_independent_seeds_per_scale") == 9
            and extension.get("outcome_dependent_early_stopping") is False,
            "Paper-grade completion or primary causal gate drifted.",
        )
        _require(
            "evidence only against the tested" in payload.get("claim_boundary", "")
            and "pilot Tier-S workloads" in payload.get("claim_boundary", ""),
            "Paper-grade pilot claim boundary drifted.",
        )
    elif name == "p2_seed_extension":
        blinding = payload.get("outcome_blinding", {})
        primary = payload.get("primary_cohort", {})
        extension = payload.get("extension_cohort", {})
        inference = payload.get("combined_confirmatory_inference", {})
        volume = payload.get("planned_extension_volume", {})
        execution = payload.get("execution_contract", {})
        _require(
            payload.get("status") == "preregistered_before_primary_outcome_inspection"
            and blinding.get("primary_outcome_summary_inspected") is False
            and primary.get("training_seeds")
            == [6071401, 6071402, 6071403, 6071404, 6071405]
            and primary.get("artifacts_are_immutable") is True
            and extension.get("training_seeds") == [6071406, 6071407, 6071408, 6071409]
            and extension.get("calibration_seeds")
            == [7071406, 7071407, 7071408, 7071409]
            and extension.get("evaluation_seeds")
            == [8071406, 8071407, 8071408, 8071409]
            and set(primary.get("training_seeds", ())).isdisjoint(
                extension.get("training_seeds", ())
            )
            and extension.get("scales") == ["s55", "s151"]
            and extension.get("outcome_dependent_early_stopping") is False
            and extension.get("separate_artifact_namespace_required") is True,
            "P2 seed-extension cohort boundary drifted.",
        )
        _require(
            inference.get("independent_training_seeds_per_scale") == 9
            and inference.get("exact_sign_assignments") == 512
            and inference.get("minimum_attainable_two_sided_seed_p") == 0.00390625
            and inference.get("minimum_attainable_holm_adjusted_family_p") == 0.03515625
            and volume.get("core_shards") == 3_600
            and volume.get("causal_shards") == 7_200
            and execution.get("primary_gate", {}).get("required_unique_shards") == 4_500
            and execution.get("primary_causal_gate", {}).get("required_unique_shards")
            == 9_000
            and execution.get("per_checkpoint_equivalence_required") is True
            and len(execution.get("commands", [])) == 8,
            "P2 seed-extension completion or inference boundary drifted.",
        )
        failure_rules = payload.get("failure_rules", [])
        _require(
            any("Do not stop the extension" in rule for rule in failure_rules)
            and any("Do not replace a failed primary seed" in rule for rule in failure_rules),
            "P2 seed-extension outcome-independent stopping boundary drifted.",
        )
    elif name == "p2_causal_factorial":
        primary_arms = payload.get("primary_arms", {})
        supplemental = payload.get("supplemental_baseline_arms", {})
        components = payload.get("component_contrasts", {})
        registered_arms = set(primary_arms) | set(supplemental)
        for pair in components.values():
            if isinstance(pair, list):
                registered_arms.update(pair)
        memory = payload.get("memory_matching", {})
        execution = payload.get("execution", {})
        completion = payload.get("completion_contract", {})
        gate = payload.get("central_gate", {})
        _require(
            payload.get("status") == "amended_and_frozen_before_execution"
            and payload.get("scales") == ["s55", "s151"]
            and len(payload.get("training_seeds", [])) == 5
            and len(payload.get("families", [])) == 9
            and payload.get("contexts") == [80, 128, 256, 512, 1024]
            and payload.get("primary_budget_points") == ["2x", "4x"]
            and len(registered_arms) == 16
            and set(components)
            == {
                "score_concentration",
                "temporal_reuse",
                "cross_layer_prior",
                "refresh_reuse",
                "protected_pins",
                "dense_fallback",
            },
            "P2 causal-factorial matrix boundary drifted.",
        )
        _require(
            memory.get("primary_candidate") == "calibrated+pins"
            and memory.get("primary_comparator") == "fixed+pins"
            and memory.get("maximum_relative_difference") == 0.01
            and memory.get("posthoc_accuracy_interpolation") is False
            and execution.get("total_expected_shards") == 9_000
            and execution.get("equivalence_validation", {}).get(
                "required_exact_records_total"
            )
            == 57_600
            and execution.get("component_arms_use_full_sample") is True
            and completion.get("outcome_dependent_early_stopping") is False
            and completion.get("required_primary_training_seeds_per_scale") == 5
            and completion.get("required_primary_shards") == 9_000
            and completion.get("required_extension_training_seeds_per_scale") == 4
            and completion.get("required_extension_shards") == 7_200
            and completion.get("all_registered_arms_complete_every_cell") is True,
            "P2 causal-factorial physical or completion boundary drifted.",
        )
        _require(
            payload.get("statistics", {}).get("contrast_correction", "").startswith(
                "Holm-Bonferroni over all 15 preregistered contrasts"
            )
            and gate.get("contrast") == "calibrated+pins - fixed+pins"
            and gate.get("required_budget_points") == ["2x", "4x"]
            and "all five strictly positive" in gate.get("seed_effects", "")
            and "within 1 percent" in gate.get("memory_match", "")
            and "Tier-S synthetic workloads only" in payload.get("claim_boundary", ""),
            "P2 causal-factorial central claim boundary drifted.",
        )
    elif name == "online_learned_lookahead":
        claim = payload.get("claim_boundary", {})
        causal = payload.get("causal_contract", {})
        splits = payload.get("splits", {})
        matrix = payload.get("matrix", {})
        gate = payload.get("positive_gate", {})
        failures = payload.get("failure_rules", {})
        _require(
            payload.get("status") == "frozen_before_execution"
            and "separate exploratory controller study" in payload.get("role", "")
            and "excluded from the preregistered P2 primary causal gate"
            in payload.get("role", "")
            and "same-token causality" in claim.get("ineligible", [])
            and "replacement of a failed preregistered P2 primary arm"
            in claim.get("ineligible", [])
            and causal.get("applied_time") == "token t+1 only"
            and "forbidden" in causal.get("future_information", "")
            and "inside the active physical hot-memory budget"
            in causal.get("protected_pins", ""),
            "P1 online-lookahead causal boundary drifted.",
        )
        _require(
            splits.get("training_seed_namespace") == 110714000
            and splits.get("calibration_seed_namespace") == 120714000
            and splits.get("test_seed_namespace") == 130714000
            and "disjoint" in splits.get("overlap_policy", "")
            and "no retuning" in splits.get("test_access", "")
            and matrix.get("scales") == ["s55", "s151"]
            and matrix.get("budgets") == ["2x", "4x"]
            and len(matrix.get("workload_families", [])) == 9
            and len(payload.get("arms", [])) == 6,
            "P1 online-lookahead split or matrix boundary drifted.",
        )
        _require(
            "without higher measured hot HBM or useful transfer bytes"
            in gate.get("pareto", "")
            and gate.get("budget") == "zero logical or physical budget violations"
            and "all cache lifecycle and replay digests verify" in gate.get("replay", "")
            and failures.get("missing_scale_or_seed") == "unverified, never success"
            and payload.get("execution_equivalence", {}).get(
                "required_scale_seed_probes"
            )
            == 10
            and "only after the frozen P2 causal summary is terminal"
            in payload.get("execution_order", ""),
            "P1 online-lookahead Pareto, replay, or sequence boundary drifted.",
        )
    elif name == "cross_family":
        relationship = payload.get("relationship_to_primary", {})
        sequence = payload.get("sequence_gate", {})
        benchmark = payload.get("benchmark", {})
        statistics = payload.get("statistics", {})
        gate = payload.get("transfer_gate", {})
        model = payload.get("model", {})
        _require(
            payload.get("status") == "frozen_before_any_p3_model_prediction"
            and relationship.get("transfer_model_family") == "Phi-4"
            and relationship.get("pooled_with_primary") is False
            and relationship.get("outcome_dependent_execution") is False
            and relationship.get("phi_specific_tuning_allowed") is False
            and model.get("repo_id") == "microsoft/Phi-4-mini-instruct"
            and model.get("revision") == "cfbefacb99257ffa30c83adab238a50856ac3083"
            and "terminal nine-seed P2 evidence" in sequence.get("policy", ""),
            "P3 cross-family sequence or transfer boundary drifted.",
        )
        _require(
            benchmark.get("lengths_tokens") == [8192, 32768, 131072]
            and benchmark.get("tasks_per_length") == 13
            and benchmark.get("samples_per_task_length") == 100
            and benchmark.get("predictions_per_arm") == 3_900
            and benchmark.get("paired_predictions_total") == 7_800
            and statistics.get("paired_bootstrap_seed") == 9_171_501
            and gate.get("maximum_realized_kv_fraction") == 0.51
            and "cannot trigger Phi-specific method selection"
            in gate.get("interpretation", "")
            and "not a second full natural-language suite"
            in payload.get("claim_boundary", ""),
            "P3 cross-family matrix or claim boundary drifted.",
        )
    elif name == "p3_ruler":
        amendments = payload.get("amendments", [])
        observed = payload.get("sequence_gate", {}).get("observed_before_gate", {})
        _require(
            payload.get("status") == "amended_and_frozen_before_execution"
            and isinstance(amendments, list)
            and len(amendments) == 4,
            "P3 RULER pre-execution amendment record drifted.",
        )
        _require(
            observed
            == {
                "complete_dataset_manifests": 0,
                "orphan_deterministic_task_files": 1,
                "orphan_rows": 500,
                "result_cells": 0,
                "model_predictions": 0,
            },
            "P3 RULER pre-gate artifact accounting drifted.",
        )
        _require(
            "every model prediction require" in payload.get("sequence_gate", {}).get("policy", "")
            and "zero result cells" in amendments[1].get("reason", ""),
            "P3 RULER sequence boundary drifted.",
        )
        _require(
            "PyramidKV" in amendments[2].get("change", "")
            and "Ada-KV" in amendments[2].get("change", "")
            and payload.get("execution", {}).get("total_cells") == 57
            and payload.get("execution", {}).get("total_predictions") == 370_500,
            "P3 RULER baseline breadth drifted.",
        )
        _require(
            "pinned KVPress checkout" in amendments[3].get("change", "")
            and "site-packages" in amendments[3].get("reason", "")
            and "zero result cells" in amendments[3].get("reason", ""),
            "P3 RULER runtime import boundary drifted.",
        )
    elif name == "natural_adaptive_quota":
        benchmark = payload.get("benchmark", {})
        adaptive = payload.get("arms", {}).get("natural-adaptive-quota+pins", {})
        selection = payload.get("score_compatible_baseline_selection", {})
        amendments = payload.get("amendments", [])
        _require(
            payload.get("status") == "frozen_before_any_compatibility_arm_prediction"
            and benchmark.get("predictions_per_arm") == 32_500
            and benchmark.get("paired_predictions_total") == 65_000
            and adaptive.get("maximum_layer_adjustment_fraction") == 0.25
            and payload.get("pins", {}).get("same_budget") is True
            and selection.get("eligible_arms")
            == ["streaming_llm", "snapkv", "critical_expected_attention"]
            and isinstance(amendments, list)
            and len(amendments) == 1
            and "PyramidKV" in amendments[0].get("change", "")
            and "not an unchanged transfer" in payload.get("claim_boundary", ""),
            "P3 natural adaptive-quota boundary drifted.",
        )
    elif name == "natural_adaptive_quota_scbench":
        benchmark = payload.get("benchmark", {})
        lifecycle = payload.get("cache_lifecycle_contract", {})
        _require(
            payload.get("status") == "frozen_before_any_adaptive_scbench_prediction"
            and benchmark.get("predictions_per_arm") == 10_286
            and benchmark.get("paired_predictions_total") == 20_572
            and len(benchmark.get("modes", [])) == 2
            and len(benchmark.get("tasks", [])) == 12
            and lifecycle.get("adaptive_allocation_scope")
            == "initial shared-context prefill only"
            and lifecycle.get("same_initial_global_kept_tokens") is True
            and lifecycle.get("continuous_refresh_claim_available") is False
            and payload.get("statistics", {}).get("paired_cluster_bootstrap_seed")
            == 9_371_504
            and "does not establish continuous adaptive reallocation"
            in payload.get("claim_boundary", ""),
            "P3 adaptive SCBench boundary drifted.",
        )
    elif name == "natural_adaptive_quota_longbench_v2":
        benchmark = payload.get("benchmark", {})
        lifecycle = payload.get("cache_lifecycle_contract", {})
        _require(
            payload.get("status")
            == "frozen_before_any_adaptive_longbench_v2_prediction"
            and benchmark.get("predictions_per_arm") == 503
            and benchmark.get("paired_predictions_total") == 1_006
            and len(benchmark.get("categories", [])) == 6
            and lifecycle.get("adaptive_allocation_scope")
            == "initial context prefill only"
            and lifecycle.get("same_initial_global_kept_tokens") is True
            and lifecycle.get("continuous_refresh_claim_available") is False
            and payload.get("statistics", {}).get("paired_bootstrap_seed")
            == 9_471_505
            and payload.get("statistics", {}).get("holm_family_size") == 6
            and "does not establish continuous adaptive reallocation"
            in payload.get("claim_boundary", ""),
            "P3 adaptive LongBench-v2 boundary drifted.",
        )
    elif name == "natural_adaptive_quota_longmemeval":
        benchmark = payload.get("benchmark", {})
        lifecycle = payload.get("cache_lifecycle_contract", {})
        metric = payload.get("official_metric_contract", {})
        failures = payload.get("failure_reporting", {})
        _require(
            payload.get("status")
            == "frozen_before_any_adaptive_longmemeval_generation"
            and benchmark.get("examples_per_arm") == 500
            and benchmark.get("paired_predictions_total") == 1_000
            and lifecycle.get("adaptive_allocation_scope")
            == "initial history-context prefill only"
            and lifecycle.get("same_initial_global_kept_tokens") is True
            and lifecycle.get("continuous_refresh_claim_available") is False
            and metric.get("judge_model") == "gpt-4o-2024-08-06"
            and metric.get("initial_execution_mode") == "blocked"
            and metric.get("quality_classification_before_official_judging")
            == "unverified"
            and metric.get("confirmation_gate_available_before_official_judging")
            is False
            and "no deterministic" in metric.get("auxiliary_metric_policy", "")
            and "separately frozen manifest" in metric.get("paid_judge_policy", "")
            and "judge-blocked" in failures.get("allowed_failure_types", [])
            and "no model-quality success"
            in payload.get("claim_boundary", ""),
            "P3 adaptive LongMemEval boundary drifted.",
        )
    elif name == "natural_adaptive_quota_mrcr":
        benchmark = payload.get("benchmark", {})
        lifecycle = payload.get("cache_lifecycle_contract", {})
        _require(
            payload.get("status") == "frozen_before_any_adaptive_mrcr_prediction"
            and benchmark.get("predictions_per_arm") == 1_500
            and benchmark.get("paired_predictions_total") == 3_000
            and benchmark.get("needle_counts") == [2, 4, 8]
            and benchmark.get("primary_token_bins") == 5
            and lifecycle.get("adaptive_allocation_scope")
            == "initial context prefill only"
            and lifecycle.get("same_initial_global_kept_tokens") is True
            and lifecycle.get("continuous_refresh_claim_available") is False
            and payload.get("statistics", {}).get("paired_bootstrap_seed")
            == 9_571_506
            and payload.get("statistics", {}).get("holm_family_size") == 15
            and "does not establish continuous adaptive reallocation"
            in payload.get("claim_boundary", ""),
            "P3 adaptive MRCR boundary drifted.",
        )
    elif name == "natural_adaptive_quota_suite":
        coverage = payload.get("coverage", {})
        statistics = payload.get("statistics", {})
        gate = payload.get("confirmation_gate", {})
        components = payload.get("components", {})
        _require(
            payload.get("status") == "frozen_before_any_adaptive_suite_summary"
            and tuple(components) == ("RULER", "SCBench", "LongBench-v2", "MRCR")
            and coverage.get("benchmarks") == 4
            and coverage.get("predictions_per_arm") == 44_789
            and coverage.get("paired_predictions_total") == 89_578
            and "official GPT-4o judge unavailable"
            in coverage.get("excluded_from_adaptive_suite", {}).get("LongMemEval", "")
            and statistics.get("no_cross_benchmark_score_pooling") is True
            and statistics.get("no_cross_benchmark_p_value_pooling") is True
            and statistics.get("suite_gate_uses_component_p_values") is False
            and statistics.get("outcome_dependent_benchmark_selection") is False
            and gate.get("required_terminal_components") == 4
            and gate.get("minimum_component_confirmation_gates_passed") == 3
            and gate.get("minimum_nonnegative_overall_effects") == 4
            and gate.get("all_required_physical_and_failure_checks_must_pass") is True
            and "does not pool incomparable benchmark scores"
            in payload.get("claim_boundary", ""),
            "P3 adaptive natural suite boundary drifted.",
        )
    elif name == "cross_family_adaptive_quota":
        benchmark = payload.get("benchmark", {})
        relationship = payload.get("relationship_to_other_cohorts", {})
        physical = payload.get("physical_contract", {})
        _require(
            payload.get("status") == "frozen_before_any_cross_family_adaptive_prediction"
            and benchmark.get("predictions_per_arm") == 3_900
            and benchmark.get("paired_predictions_total") == 7_800
            and relationship.get("pooled_with_qwen") is False
            and relationship.get("phi_specific_tuning_or_reselection_allowed") is False
            and physical.get("same_global_kept_tokens") is True
            and payload.get("statistics", {}).get("paired_bootstrap_seed") == 9_271_503
            and "not unchanged transfer" in payload.get("claim_boundary", ""),
            "P3 cross-family adaptive-quota boundary drifted.",
        )
    elif name == "cross_family_adaptive_quota_longbench_v2":
        benchmark = payload.get("benchmark", {})
        lifecycle = payload.get("cache_lifecycle_contract", {})
        relationship = payload.get("relationship_to_other_cohorts", {})
        model = payload.get("model", {})
        scorer_selection = payload.get("scorer_selection", {})
        statistics = payload.get("statistics", {})
        _require(
            payload.get("status")
            == "frozen_before_any_phi_longbench_v2_prediction"
            and benchmark.get("predictions_per_arm") == 503
            and benchmark.get("paired_predictions_total") == 1_006
            and len(benchmark.get("categories", [])) == 6
            and model.get("revision") == "cfbefacb99257ffa30c83adab238a50856ac3083"
            and model.get("num_hidden_layers") == 32
            and model.get("maximum_supported_context_tokens") == 131_072
            and lifecycle.get("adaptive_allocation_scope")
            == "initial context prefill only"
            and lifecycle.get("same_initial_global_kept_tokens") is True
            and lifecycle.get("continuous_refresh_claim_available") is False
            and relationship.get("pooled_with_qwen_longbench_v2") is False
            and relationship.get("pooled_with_phi_ruler") is False
            and relationship.get("phi_specific_tuning_or_reselection_allowed") is False
            and scorer_selection.get("phi_specific_reselection_allowed") is False
            and statistics.get("paired_bootstrap_seed") == 9_671_507
            and statistics.get("paired_bootstrap_resamples") == 10_000
            and statistics.get("holm_family_size") == 6
            and statistics.get("secondary_slices_are_descriptive") is True
            and "one held-out Phi-4-mini checkpoint"
            in payload.get("claim_boundary", ""),
            "P3 cross-family adaptive-quota LongBench-v2 boundary drifted.",
        )
    elif name == "natural_suite":
        baselines = payload.get("external_baselines", {})
        kvpress = baselines.get("kvpress", {})
        flashmemory = baselines.get("FlashMemory-DeepSeek-V4", {})
        indexcache = baselines.get("IndexCache", {})
        indexcache_files = indexcache.get("files_sha256", {})
        _require(
            payload.get("status") == "amended_and_frozen_before_execution"
            and kvpress.get("revision") == "6d965557a5b9f0201a2301b23c454473dd681d0d",
            "P3 compatible-model baseline boundary drifted.",
        )
        _require(
            tuple(kvpress.get("compatible_qwen3_methods", ()))
            == (
                "streaming-llm",
                "snapkv",
                "pyramidkv",
                "adakv-snapkv",
                "expected-attention",
                "critical-expected-attention",
            ),
            "P3 compatible-model baseline coverage drifted.",
        )
        _require(
            "not a claim that the selected algorithm uses fixed allocation"
            in kvpress.get("fixed_baseline_selection", {}).get("label_semantics", ""),
            "P3 compatible baseline legacy-label boundary drifted.",
        )
        _require(
            "no winner-vs-runner-up significance claim"
            in kvpress.get("fixed_baseline_selection", {}).get("selection_inference_boundary", "")
            and "held-out transfer"
            in kvpress.get("fixed_baseline_selection", {}).get("selection_inference_boundary", ""),
            "P3 compatible baseline selection-inference boundary drifted.",
        )
        _require(
            flashmemory.get("manifest")
            == "research/adaptive_v4_memory/manifests/p3-flashmemory-deepseek-v4-v1.json"
            and flashmemory.get("compatible_with_primary_qwen3_model") is False
            and "never project" in flashmemory.get("action", ""),
            "FlashMemory architecture boundary drifted.",
        )
        _require(
            indexcache.get("repository") == "https://github.com/THUDM/IndexCache"
            and indexcache.get("revision") == "08d22d69b1aa2aa0a3de23df6d6b88dbd5b5d044"
            and indexcache.get("license") == "apache-2.0"
            and indexcache.get("compatible_with_primary_qwen3_model") is False
            and "DeepSeek Sparse Attention" in indexcache.get("supported_architecture_boundary", "")
            and "supported DSA runtime" in indexcache.get("action", "")
            and indexcache_files
            == {
                "LICENSE": "7ecd8ce1d30b8aa26232f5c7c878cca53bc273547ce52f1678f955154746e64f",
                "README.md": "b925cec609200200d1f5d076083589d3b19fb78a32b9f5c541db09baf70b5b49",
                "indexcache.patch": "aa0e78e6e7ffd25e9d98f45fd608dc3547af937e5f77e01b5c01f1aaaf5df5b7",
                "indexcache_vllm.patch": "2044f07ecd65b2bebe0b770bbc3f0246785b6eb5026311345722ef7d271ce356",
            },
            "IndexCache architecture or provenance boundary drifted.",
        )
    elif name == "p4_500k_context":
        amendments = payload.get("protocol_amendments", [])
        execution = payload.get("execution", {})
        _require(
            payload.get("status") == "amended_and_frozen_before_execution"
            and isinstance(amendments, list)
            and len(amendments) == 1,
            "P4 500K pre-execution amendment record drifted.",
        )
        _require(
            execution.get("maximum_cell_timeout_seconds") == 21_600
            and "uninterruptible native CUDA" in execution.get("timeout_enforcement", ""),
            "P4 500K timeout claim boundary drifted.",
        )
    elif name == "experiment_scale_audit":
        _validate_experiment_scale_audit(payload)
    elif name == "official_deepseek_v4":
        blockers = payload.get("blockers")
        blocker_ids = (
            {row.get("id") for row in blockers if isinstance(row, dict)}
            if isinstance(blockers, list)
            else set()
        )
        modes = payload.get("runtime_modes", {})
        mode_a = modes.get("mode_a_score_masking", {})
        mode_b = modes.get("mode_b_pd_disaggregated", {})
        audit = payload.get("public_release_contract_audit", {})
        verification = payload.get("post_acquisition_verification", {})
        protocol = payload.get("execution_protocol", {})
        upstream = payload.get("upstream", {})
        base_model = upstream.get("base_model", {})
        resources = payload.get("local_resource_audit", {})
        cost = payload.get("cost_proxy", {})
        mode_a_cost = cost.get("mode_a", {})
        mode_b_cost = cost.get("mode_b", {})
        acquisition = payload.get("frozen_acquisition_commands")
        _require(
            payload.get("status") == "blocked_before_execution",
            "Official DeepSeek-V4 boundary no longer fails closed.",
        )
        _require(
            "not_an_executed_result" in payload.get("evidence_tier", ""),
            "Official DeepSeek-V4 boundary could be misread as executed evidence.",
        )
        _require(
            mode_a.get("full_kv_remains_on_gpu") is True,
            "Official DeepSeek-V4 Mode A memory boundary drifted.",
        )
        _require(
            mode_b.get("total_accelerator_slots") == 16,
            "Official DeepSeek-V4 Mode B topology boundary drifted.",
        )
        _require(
            mode_a.get("launch")
            == "MODEL=/models/deepseek-v4-flash CKPT=/weights/top3_R930_joint.pt TP=4 bash start_server.sh"
            and mode_b.get("startup_order")
            == [
                "TGT_CONC=60 TGT_CTX=524288 CTX_LEN=1100000 MODEL=/models/deepseek-v4-flash CKPT=/weights/top3_R930_joint.pt bash high_concurrency/launch_decode.sh",
                "CTX_LEN=1100000 SWA_RATIO=0.1 HOST=<PREFILL_IP> MODEL=/models/deepseek-v4-flash bash high_concurrency/launch_prefill.sh",
                "PREFILL_IP=<PREFILL_IP> DECODE_IP=<DECODE_IP> bash high_concurrency/launch_router.sh",
            ],
            "Official DeepSeek-V4 launch sequence drifted.",
        )
        _require(
            base_model.get("revision") == "60d8d70770c6776ff598c94bb586a859a38244f1"
            and base_model.get("safetensors_files") == 46
            and base_model.get("safetensors_bytes") == 159_617_149_040,
            "Official DeepSeek-V4 base-weight storage contract drifted.",
        )
        _require(
            type(resources.get("physical_memory_bytes")) is int
            and type(resources.get("available_memory_plus_swap_bytes")) is int
            and type(resources.get("disk_free_bytes")) is int
            and resources["physical_memory_bytes"] < base_model["safetensors_bytes"]
            and resources["available_memory_plus_swap_bytes"] < base_model["safetensors_bytes"]
            and resources["disk_free_bytes"] > base_model["safetensors_bytes"]
            and resources.get("base_weights_fit_physical_memory") is False
            and resources.get("base_weights_fit_available_memory_plus_swap") is False
            and resources.get("base_weights_fit_disk") is True
            and resources.get("mode_a_accelerator_topology_available") is False
            and resources.get("mode_b_accelerator_topology_available") is False,
            "Official DeepSeek-V4 local resource audit drifted.",
        )
        _require(
            cost.get("currency") == "USD"
            and cost.get("provider") == "Lambda"
            and "planning proxy" in cost.get("warning", "")
            and mode_a_cost.get("accelerator_slots") == 4
            and mode_b_cost.get("accelerator_slots") == 16
            and mode_a_cost.get("estimated_usd_per_hour")
            == 4 * cost.get("h100_sxm_price_per_gpu_hour", {}).get("four_gpu_instance", 0)
            and mode_b_cost.get("estimated_usd_per_hour")
            == 16 * cost.get("h100_sxm_price_per_gpu_hour", {}).get("eight_gpu_instance", 0)
            and mode_a_cost.get("estimated_usd_for_24_hours")
            == 24 * mode_a_cost.get("estimated_usd_per_hour", 0)
            and mode_b_cost.get("estimated_usd_for_24_hours")
            == 24 * mode_b_cost.get("estimated_usd_per_hour", 0)
            and mode_b_cost.get("network_and_storage_surcharges_included") is False,
            "Official DeepSeek-V4 cost contract drifted.",
        )
        _require(
            acquisition
            == [
                "git clone https://github.com/libertywing/FlashMemory-Deepseek-V4.git",
                "git -C FlashMemory-Deepseek-V4 checkout 39fe54def633496cb2b1bd44898135e3547058b3",
                "hf download deepseek-ai/DeepSeek-V4-Flash --revision 60d8d70770c6776ff598c94bb586a859a38244f1 --local-dir /models/deepseek-v4-flash",
                "hf download libertywing/FlashMemory-Deepseek-V4 --revision 70431ba57bfcce00ffd9d0174aed1b8ca5c32a2e --local-dir /weights/flashmemory-public",
            ],
            "Official DeepSeek-V4 acquisition contract drifted.",
        )
        _require(
            audit.get("pt_checkpoint_present_in_published_hf_snapshot") is False
            and audit.get("documented_safetensors_to_serving_conversion_present") is False,
            "Official DeepSeek-V4 checkpoint blocker drifted.",
        )
        _require(
            blocker_ids
            >= {
                "checkpoint-runtime-contract",
                "local-memory-capacity",
                "accelerator-topology",
            },
            "Official DeepSeek-V4 hard blockers are incomplete.",
        )
        _require(
            verification.get("serving_checkpoint", {}).get("current_status") == "unavailable"
            and len(verification.get("serving_checkpoint", {}).get("required_before_execution", []))
            >= 4,
            "Official DeepSeek-V4 checkpoint verification contract is incomplete.",
        )
        systems = protocol.get("mode_b_physical_systems", {})
        quality = protocol.get("mode_a_quality_only", {})
        _require(
            systems.get("context_tokens") == [8192, 32768, 131072, 512000]
            and systems.get("batch_sizes") == [1, 4, 8, 16]
            and systems.get("concurrency") == [1, 8, 32]
            and systems.get("generation_tokens") == [128, 512, 2048]
            and systems.get("warmups_per_cell") == 5
            and systems.get("timed_repetitions_per_cell") == 30,
            "Official DeepSeek-V4 systems execution matrix drifted.",
        )
        _require(
            quality.get("benchmarks")
            == ["RULER", "SCBench", "LongBench-v2", "LongMemEval", "MRCR"]
            and len(systems.get("required_metrics", [])) >= 10
            and len(protocol.get("paired_invariants", [])) >= 5
            and len(protocol.get("artifact_contract", [])) >= 4,
            "Official DeepSeek-V4 execution evidence contract is incomplete.",
        )
        preconditions = payload.get("execution_preconditions", [])
        failure_policy = protocol.get("failure_policy", "")
        _require(
            len(preconditions) >= 5
            and any("top3_R930_joint.pt" in item for item in preconditions)
            and any("golden fixture" in item for item in preconditions)
            and all(term in failure_policy for term in ("complete", "partial", "failed"))
            and "may not be silently retried" in failure_policy,
            "Official DeepSeek-V4 execution preconditions or failure policy drifted.",
        )
    elif name == "production_runtime_blocker":
        _require(
            payload.get("status") == "external-fused-dynamic-runtime-unavailable",
            "External production runtime boundary no longer fails closed.",
        )
        _require(
            "never relabel" in payload.get("failure_policy", ""),
            "External production runtime anti-relabel policy is missing.",
        )
        _require(
            "multi-GPU" in payload.get("claim_boundary", ""),
            "External production runtime claim boundary is incomplete.",
        )
    return payload


def _clean_source() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return commit, dirty


def _validate_evidence(name: str, path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    payload = _load(path)
    _require(
        payload.get("experiment_id") == contract["experiment_id"],
        f"Wrong {name} experiment id.",
    )
    _require(payload.get("source", {}).get("dirty") is False, f"Dirty {name} evidence.")
    audit = payload.get("audit", {})
    for field, expected in contract["required_audit"].items():
        _require(audit.get(field) == expected, f"{name} audit field {field} drifted.")
    for section, required in contract.get("required_sections", {}).items():
        observed = payload.get(section, {})
        _require(isinstance(observed, dict), f"{name} section {section} is missing.")
        for field, expected in required.items():
            _require(
                observed.get(field) == expected,
                f"{name} section {section}.{field} drifted.",
            )
    for artifact_field in contract.get("required_artifacts", []):
        _bound_artifact(payload.get(artifact_field), f"{name} {artifact_field}")
    analysis_paths = P2_ANALYSIS_PATHS.get(name)
    if analysis_paths is not None:
        _require(
            payload.get("analysis_implementation")
            == _analysis_implementation_metadata(analysis_paths),
            f"{name} analysis implementation drifted.",
        )
    return payload


def _bound_artifact(metadata: Any, label: str) -> Path:
    _require(isinstance(metadata, dict), f"Missing {label} metadata.")
    path = Path(metadata.get("path", ""))
    _require(path.is_file(), f"Missing {label}: {path}")
    _require(metadata.get("sha256") == sha256(path), f"Digest mismatch for {label}: {path}")
    return path


def _validate_execution_audit(name: str, path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    payload = _validate_evidence(name, path, contract)
    orchestrator = Path(contract["orchestrator"])
    _require(orchestrator.is_file(), f"Missing {name} orchestrator: {orchestrator}")
    _require(
        payload.get("source", {}).get("orchestrator_sha256") == sha256(orchestrator),
        f"{name} orchestrator digest drifted.",
    )
    probes = payload.get("probes")
    _require(
        isinstance(probes, list) and len(probes) == contract["required_probes"],
        f"{name} probe coverage drifted.",
    )
    coordinates: set[tuple[Any, ...]] = set()
    for index, probe_metadata in enumerate(cast(list[Any], probes)):
        _require(isinstance(probe_metadata, dict), f"Malformed {name} probe {index}.")
        probe = probe_metadata
        child_experiment_id = contract.get("child_experiment_id")
        if child_experiment_id is not None:
            child_path = _bound_artifact(probe_metadata, f"{name} probe {index}")
            probe = _load(child_path)
            _require(
                probe.get("experiment_id") == child_experiment_id,
                f"Wrong {name} child experiment id: {child_path}",
            )
            child_artifacts = probe.get("artifacts")
            _require(
                isinstance(child_artifacts, list)
                and len(child_artifacts) == contract["child_artifacts"],
                f"{name} child artifact coverage drifted: {child_path}",
            )
            for child_index, artifact in enumerate(cast(list[Any], child_artifacts)):
                _bound_artifact(
                    artifact,
                    f"{name} child artifact {index}/{child_index}",
                )
        coordinate = tuple(probe.get(field) for field in contract["coordinate_fields"])
        _require(
            all(value is not None for value in coordinate) and coordinate not in coordinates,
            f"{name} probe coordinate coverage drifted: {coordinate}",
        )
        coordinates.add(coordinate)
        for artifact_field in contract.get("artifact_fields", []):
            _bound_artifact(
                probe.get(artifact_field),
                f"{name} {artifact_field} artifact {coordinate}",
            )
    return payload


def _validate_reproduction_guide(path: Path) -> str:
    _require(path.is_file(), f"Missing reproduction guide: {path}")
    guide = path.read_text()
    normalized_guide = " ".join(guide.split())
    missing = [marker for marker in REPRODUCTION_REQUIRED_MARKERS if marker not in normalized_guide]
    _require(not missing, f"Reproduction guide is incomplete: {missing}")
    _require(
        "resume-safe" in normalized_guide
        and "Never delete a terminal failure artifact" in normalized_guide
        and "must not be reported as passed" in normalized_guide
        and "outside the completion gate" in normalized_guide,
        "Reproduction failure and CI boundaries drifted.",
    )
    return guide


def _traceability_rows(
    payload: dict[str, Any],
    package_manifest: dict[str, Any],
    classifications: dict[str, str],
) -> list[dict[str, Any]]:
    _require(
        payload.get("experiment_id") == "adaptive-v4-memory-p5-requirement-traceability-v1",
        "Wrong P5 requirement traceability manifest.",
    )
    raw_contracts = payload.get("verification_contracts")
    _require(isinstance(raw_contracts, dict), "Missing traceability verification contracts.")
    contracts = cast(dict[str, Any], raw_contracts)
    raw_release = contracts.get("final-local-release-gate")
    _require(isinstance(raw_release, dict), "Missing final local release-gate contract.")
    release = cast(dict[str, Any], raw_release)
    _require(
        release.get("commands") == FINAL_RELEASE_COMMANDS
        and release.get("github_actions") == "disabled_by_user_not_required"
        and release.get("github_actions_current_status") == "disabled_by_user"
        and release.get("github_actions_passed") is False
        and release.get("github_actions_required_for_completion") is False
        and release.get("runner") == "research/adaptive_v4_memory/scripts/run_p5_release_gate.py"
        and release.get("output")
        == "artifacts/adaptive_v4_memory/paper_grade/p5/local-release-gate.summary.json"
        and release.get("timing") == "after-final-paper-package-generation",
        "Final local release-gate contract drifted.",
    )
    _require(
        release.get("source_remote_sync")
        == {
            "required": True,
            "upstream_prefix": "origin/",
            "ahead": 0,
            "behind": 0,
            "scope": "local origin tracking ref only; no fetch, PR, or CI claim",
        },
        "Final source remote-sync contract drifted.",
    )
    raw_controller = contracts.get("p1-controller-contract-tests")
    _require(isinstance(raw_controller, dict), "Missing P1 controller-contract tests.")
    controller = cast(dict[str, Any], raw_controller)
    _require(
        controller.get("timing") == "within-final-local-release-gate"
        and controller.get("covered_by") == ".venv/bin/pytest -q"
        and controller.get("tests") == CONTROLLER_CONTRACT_TESTS
        and controller.get("covered_by") in release.get("commands", []),
        "P1 controller-contract verification drifted.",
    )
    for contract_name, node_id in CONTROLLER_CONTRACT_TESTS.items():
        path_text, separator, function_name = node_id.partition("::")
        path = Path(path_text)
        _require(
            separator == "::" and path.is_file(),
            f"Missing P1 {contract_name} test source: {path}",
        )
        tree = ast.parse(path.read_text(), filename=str(path))
        _require(
            any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function_name
                for node in tree.body
            ),
            f"Missing P1 {contract_name} test node: {node_id}",
        )
    raw_requirements = payload.get("requirements")
    _require(isinstance(raw_requirements, list), "Missing traceability requirements.")
    requirements = cast(list[Any], raw_requirements)
    ids = [row.get("id") for row in requirements if isinstance(row, dict)]
    _require(
        len(ids) == len(REQUIRED_TRACEABILITY_IDS)
        and len(set(ids)) == len(ids)
        and set(ids) == REQUIRED_TRACEABILITY_IDS,
        "P0-P5 completion traceability coverage drifted.",
    )
    evidence_names = set(cast(dict[str, Any], package_manifest["evidence"]))
    execution_names = set(cast(dict[str, Any], package_manifest["execution_audits"]))
    boundary_names = set(cast(dict[str, Any], package_manifest["boundary_manifests"]))
    generated_names = set(cast(list[str], package_manifest["generated_files"]))
    contract_names = set(contracts)
    known = {
        "evidence": evidence_names,
        "execution-audit": execution_names,
        "boundary-manifest": boundary_names,
        "generated-output": generated_names,
        "generator": {"paper_package_generator"},
        "verification-contract": contract_names,
    }
    rows: list[dict[str, Any]] = []
    for raw_requirement in requirements:
        _require(isinstance(raw_requirement, dict), "Malformed traceability requirement.")
        requirement = cast(dict[str, Any], raw_requirement)
        requirement_id = requirement.get("id")
        _require(isinstance(requirement_id, str), "Malformed traceability requirement id.")
        requirement_id = cast(str, requirement_id)
        expected_phase = "completion" if requirement_id.startswith("C.") else requirement_id[:2]
        _require(
            requirement.get("phase") == expected_phase,
            f"Traceability phase drifted for {requirement_id}.",
        )
        description = requirement.get("requirement")
        _require(
            isinstance(description, str) and len(description.strip()) >= 20,
            f"Traceability requirement text is incomplete for {requirement_id}.",
        )
        description = cast(str, description)
        raw_sources = requirement.get("sources")
        _require(
            isinstance(raw_sources, list) and len(raw_sources) > 0,
            f"Traceability sources are missing for {requirement_id}.",
        )
        sources = cast(list[Any], raw_sources)
        seen_sources: set[tuple[str, str]] = set()
        for raw_source in sources:
            _require(isinstance(raw_source, dict), f"Malformed source for {requirement_id}.")
            source = cast(dict[str, Any], raw_source)
            raw_kind = source.get("kind")
            raw_name = source.get("name")
            _require(
                isinstance(raw_kind, str)
                and raw_kind in TRACEABILITY_SOURCE_KINDS
                and isinstance(raw_name, str),
                f"Unknown traceability source for {requirement_id}.",
            )
            kind = cast(str, raw_kind)
            name = cast(str, raw_name)
            coordinate = (kind, name)
            _require(
                name in known[kind] and coordinate not in seen_sources,
                f"Unbound or duplicate traceability source for {requirement_id}: {coordinate}",
            )
            seen_sources.add(coordinate)
            if kind == "evidence":
                binding_status = "digest-bound-evidence"
                scientific_classification = classifications[name]
            elif kind == "execution-audit":
                binding_status = "validated-execution-audit"
                scientific_classification = "not-applicable"
            elif kind == "boundary-manifest":
                binding_status = "validated-claim-boundary"
                scientific_classification = classifications.get(name, "not-applicable")
            elif kind == "generated-output":
                binding_status = "declared-digest-bound-output"
                scientific_classification = "not-applicable"
            elif kind == "generator":
                _paper_package_generator_input()
                binding_status = "digest-bound-generator"
                scientific_classification = "not-applicable"
            else:
                binding_status = "scheduled-final-verification"
                scientific_classification = "not-applicable"
            rows.append(
                {
                    "requirement_id": requirement_id,
                    "phase": expected_phase,
                    "requirement": description,
                    "source_kind": kind,
                    "source_name": name,
                    "binding_status": binding_status,
                    "scientific_classification": scientific_classification,
                }
            )
    return rows


def _classify_500k_preflight(payload: dict[str, Any]) -> str:
    audit = payload["audit"]
    successes = audit.get("successful_policy_attempts")
    failures = audit.get("failed_policy_attempts")
    terminal = (
        audit.get("all_terminal_cells_verified") is True
        and audit.get("all_artifact_digests_verified") is True
        and audit.get("context_tokens") == 500_000
        and audit.get("generation_tokens") == 128
        and audit.get("scales_attempted") == 2
        and audit.get("terminal_policy_attempts") == 4
        and type(successes) is int
        and type(failures) is int
        and successes >= 0
        and failures >= 0
        and successes + failures == 4
        and audit.get("whole_cell_timeout_contract_verified") is True
        and audit.get("performance_claim_available") is False
    )
    if not terminal:
        return "unverified"
    return "bounded-result" if successes > 0 else "negative-result"


def _classify_validated_confirmatory_core(payload: dict[str, Any]) -> str:
    quality_gate = payload.get("quality_gate")
    if not isinstance(quality_gate, list) or not quality_gate:
        return "unverified"
    return (
        "success"
        if any(
            isinstance(row, dict) and row.get("passes_fixed_baseline_component") is True
            for row in quality_gate
        )
        else "negative-result"
    )


def _validated_confirmatory_causal_pareto(payload: dict[str, Any]) -> tuple[bool, str]:
    gate = payload.get("primary_causal_gate")
    paired = payload.get("paired_statistics")
    physical = payload.get("physical_hot_memory")
    _require(
        isinstance(gate, dict)
        and isinstance(paired, dict)
        and isinstance(physical, dict),
        "Confirmatory causal Pareto evidence is incomplete.",
    )
    gate = cast(dict[str, Any], gate)
    paired = cast(dict[str, Any], paired)
    physical = cast(dict[str, Any], physical)
    contrast = paired.get("adaptive_quota_with_pins")
    _require(isinstance(contrast, dict), "Primary causal contrast is missing.")
    contrast = cast(dict[str, Any], contrast)
    quality_cells = contrast.get("cells")
    seed_cells = contrast.get("by_seed")
    memory_cells = physical.get("aggregate")
    gate_cells = gate.get("cells")
    _require(
        all(isinstance(rows, list) for rows in (quality_cells, seed_cells, memory_cells, gate_cells)),
        "Confirmatory causal Pareto cells are missing.",
    )
    assert isinstance(quality_cells, list)
    assert isinstance(seed_cells, list)
    assert isinstance(memory_cells, list)
    assert isinstance(gate_cells, list)
    expected = {(scale, budget) for scale in ("s55", "s151") for budget in ("2x", "4x")}

    def coordinates(rows: list[Any]) -> set[tuple[Any, Any]]:
        return {
            (row.get("scale"), row.get("budget"))
            for row in rows
            if isinstance(row, dict)
        }

    _require(
        len(quality_cells) == len(memory_cells) == len(gate_cells) == 4
        and coordinates(quality_cells) == coordinates(memory_cells) == coordinates(gate_cells)
        == expected,
        "Confirmatory causal Pareto coordinate coverage drifted.",
    )
    seeds = payload.get("audit", {}).get("independent_seed_clusters_per_cell")
    _require(type(seeds) is int and seeds >= 5, "Confirmatory causal seed count drifted.")
    assert type(seeds) is int
    recomputed_cells: list[bool] = []
    for scale, budget in sorted(expected):
        quality = next(
            row for row in quality_cells if row["scale"] == scale and row["budget"] == budget
        )
        seed_rows = [
            row for row in seed_cells if row.get("scale") == scale and row.get("budget") == budget
        ]
        memory = next(
            row for row in memory_cells if row["scale"] == scale and row["budget"] == budget
        )
        observed = next(
            row for row in gate_cells if row["scale"] == scale and row["budget"] == budget
        )
        interval = quality.get("four_cell_corrected_bootstrap", {}).get(
            "confidence_interval"
        )
        _require(
            isinstance(interval, list)
            and len(interval) == 2
            and all(isinstance(value, (int, float)) for value in interval)
            and len(seed_rows) == seeds
            and all(isinstance(row.get("mean_difference"), (int, float)) for row in seed_rows)
            and isinstance(quality.get("mean_difference"), (int, float))
            and isinstance(memory.get("relative_difference"), (int, float))
            and isinstance(memory.get("all_seed_cells_within_one_percent"), bool),
            f"Confirmatory causal Pareto schema drifted for {scale}/{budget}.",
        )
        positive_seeds = sum(row["mean_difference"] > 0.0 for row in seed_rows)
        expected_fields = {
            "pooled_effect_positive": quality["mean_difference"] > 0.0,
            "four_cell_corrected_lower_bound": interval[0],
            "four_cell_corrected_lower_bound_positive": interval[0] > 0.0,
            "positive_seed_effects": positive_seeds,
            "required_seed_effects": seeds,
            "all_seed_effects_positive": positive_seeds == seeds,
            "memory_match_relative_difference": memory["relative_difference"],
            "all_seed_memory_cells_within_one_percent": memory[
                "all_seed_cells_within_one_percent"
            ],
        }
        cell_passed = all(
            (
                expected_fields["pooled_effect_positive"],
                expected_fields["four_cell_corrected_lower_bound_positive"],
                expected_fields["all_seed_effects_positive"],
                expected_fields["all_seed_memory_cells_within_one_percent"],
            )
        )
        _require(
            all(observed.get(field) == value for field, value in expected_fields.items())
            and observed.get("passed") is cell_passed,
            f"Confirmatory causal Pareto gate drifted for {scale}/{budget}.",
        )
        recomputed_cells.append(cell_passed)
    passed = all(recomputed_cells)
    _require(
        gate.get("candidate") == "calibrated+pins"
        and gate.get("comparator") == "fixed+pins"
        and gate.get("scales") == ["s55", "s151"]
        and gate.get("budgets") == ["2x", "4x"]
        and gate.get("seeds_per_scale") == seeds
        and gate.get("required_cells") == 4
        and gate.get("passed") is passed,
        "Confirmatory causal top-level Pareto gate drifted.",
    )
    verdict = (
        "established for calibrated+pins over fixed+pins"
        if passed
        else "not established for the tested calibrated+pins controller"
    )
    return passed, verdict


def _classify_validated_confirmatory_causal(payload: dict[str, Any]) -> str:
    passed, _verdict = _validated_confirmatory_causal_pareto(payload)
    return "success" if passed else "bounded-result"


def classify_evidence(
    p2_core: dict[str, Any],
    m5_one_token_pilot: dict[str, Any],
    m3_offline_learned_risk_pilot: dict[str, Any],
    p1_online_learned_lookahead: dict[str, Any],
    p2_causal: dict[str, Any],
    p3_ruler: dict[str, Any],
    p3_natural: dict[str, Any],
    p3_safety: dict[str, Any],
    p3_natural_safety: dict[str, Any],
    p3_ifeval: dict[str, Any],
    p3_longsafety: dict[str, Any],
    p4_500k_context: dict[str, Any],
    p4_reference_systems: dict[str, Any],
    p4_production_systems: dict[str, Any],
    p4_adaptive_systems: dict[str, Any] | None = None,
    p4_adaptive_production_systems: dict[str, Any] | None = None,
    p3_cross_family: dict[str, Any] | None = None,
    p3_cross_family_adaptive_quota: dict[str, Any] | None = None,
    p3_cross_family_adaptive_quota_longbench_v2: dict[str, Any] | None = None,
    p3_natural_adaptive_quota: dict[str, Any] | None = None,
    p3_natural_adaptive_quota_scbench: dict[str, Any] | None = None,
    p3_natural_adaptive_quota_longbench_v2: dict[str, Any] | None = None,
    p3_natural_adaptive_quota_longmemeval: dict[str, Any] | None = None,
    p3_natural_adaptive_quota_mrcr: dict[str, Any] | None = None,
    p3_natural_adaptive_quota_suite: dict[str, Any] | None = None,
) -> dict[str, str]:
    core_audit = p2_core.get("audit", {})
    core_complete = (
        core_audit.get("unique_shards") == 4_500
        and core_audit.get("all_raw_shards_verified") is True
        and core_audit.get("all_dependency_digests_verified") is True
        and core_audit.get("all_record_digests_verified") is True
        and core_audit.get("no_budget_violations") is True
        and core_audit.get("held_out_seed_contract_verified") is True
        and core_audit.get("paired_conversation_coverage_verified") is True
        and core_audit.get("execution_order_coverage_verified") is True
        and core_audit.get("exact_record_schema_verified") is True
        and core_audit.get("exact_execution_rotation_verified") is True
        and core_audit.get("exact_statistical_cell_coverage_verified") is True
        and core_audit.get("paired_units_per_seed_scale_family_context") == 200
        and core_audit.get("paired_units_per_seed_scale_family") == 1_000
        and core_audit.get("statistical_cells_per_comparison") == 1_350
        and core_audit.get("aggregate_recomputed") is True
        and core_audit.get("batch_coverage_verified") is True
        and core_audit.get("exact_seed_randomization_verified") is True
        and core_audit.get("independent_seed_clusters_per_cell") == 5
        and core_audit.get("minimum_attainable_two_sided_seed_p") == 0.0625
        and core_audit.get("seed_p_values_used_as_success_gate") is False
        and core_audit.get("family_holm_p_values_used_as_success_gate") is False
    )
    core_passed = core_complete and any(
        row.get("passes_fixed_baseline_component") is True for row in p2_core["quality_gate"]
    )
    m5_audit = m5_one_token_pilot["audit"]
    m5_complete = (
        m5_audit.get("raw_artifacts_verified") is True
        and m5_audit.get("scales_verified") == 2
        and m5_audit.get("workloads_per_scale") == 3
        and m5_audit.get("required_arms_verified") == 4
        and m5_audit.get("one_token_semantics_verified") is True
        and m5_audit.get("pilot_negative_result_verified") is True
    )
    m3_audit = m3_offline_learned_risk_pilot["audit"]
    m3_complete = (
        m3_audit.get("raw_summaries_verified") is True
        and m3_audit.get("scales_verified") == 2
        and m3_audit.get("independent_splits_verified") is True
        and m3_audit.get("train_examples_per_scale") == 768
        and m3_audit.get("calibration_examples_per_scale") == 384
        and m3_audit.get("test_examples_per_scale") == 768
        and m3_audit.get("ablation_variants_verified") == 4
        and m3_audit.get("pareto_failure_verified") is True
        and m3_audit.get("refresh_ablation_available") is False
        and m3_audit.get("offline_native_probe_semantics_verified") is True
        and m3_audit.get("online_lookahead_evidence") is False
        and m3_audit.get("implementation_sources_verified") is True
    )
    learned_audit = p1_online_learned_lookahead["audit"]
    learned_complete = (
        learned_audit.get("label_shards_verified") == 6_750
        and learned_audit.get("policies_verified") == 20
        and learned_audit.get("test_shards_verified") == 9_000
        and learned_audit.get("paired_conversations") == 180_000
        and learned_audit.get("quality_arm_conversations") == 1_080_000
        and learned_audit.get("training_seeds") == 5
        and learned_audit.get("scales") == 2
        and learned_audit.get("families") == 9
        and learned_audit.get("contexts") == 5
        and learned_audit.get("budgets") == 2
        and learned_audit.get("all_raw_digests_verified") is True
        and learned_audit.get("all_dependencies_verified") is True
        and learned_audit.get("implementation_digests_verified") is True
        and learned_audit.get("dependency_artifact_digests_verified") is True
        and learned_audit.get("checkpoint_reuse_equivalence_verified") is True
        and learned_audit.get("checkpoint_reuse_scale_seed_probes") == 10
        and learned_audit.get("exact_label_policy_test_coordinates_verified") is True
        and learned_audit.get("label_and_test_seed_schedules_verified") is True
        and learned_audit.get("train_calibration_raw_membership_and_disjointness_verified") is True
        and learned_audit.get("checkpoint_digest_consistency_verified") is True
        and learned_audit.get("raw_test_record_schema_verified") is True
        and learned_audit.get("raw_physical_metrics_and_aggregates_verified") is True
        and learned_audit.get("all_inputs_paired") is True
        and learned_audit.get("zero_budget_violations") is True
        and learned_audit.get("complete_failure_accounting") is True
        and learned_audit.get("online_token_offset_verified") is True
        and learned_audit.get("native_bootstrap_accounted") is True
        and learned_audit.get("cache_replay_contract_tested") is True
        and learned_audit.get("resolution_aware_gate_verified") is True
    )
    learned_passed = p1_online_learned_lookahead["primary_gate"].get("passed") is True
    causal_passed = p2_causal["primary_causal_gate"].get("passed") is True
    p3_complete = p3_ruler.get("benchmark_complete") is True
    natural_audit = p3_natural["audit"]
    natural_complete = (
        natural_audit.get("all_required_artifacts_verified") is True
        and natural_audit.get("all_required_baseline_cells_terminal") is True
        and natural_audit.get("all_failure_accounting_complete") is True
        and natural_audit.get("all_source_implementations_verified") is True
        and natural_audit.get("all_record_revisions_verified") is True
        and natural_audit.get("all_run_identities_verified") is True
        and natural_audit.get("all_terminal_measurement_schema_verified") is True
        and natural_audit.get("all_dataset_example_identities_verified") is True
        and natural_audit.get("all_reported_scores_recomputed_from_raw_response") is True
        and natural_audit.get("generation_seed_by_benchmark")
        == {
            "RULER": 42,
            "SCBench": 42,
            "LongBench-v2": 42,
            "LongMemEval": 42,
            "MRCR": 42,
        }
        and natural_audit.get("all_paired_quality_contrasts_verified") is True
        and natural_audit.get("dataset_license_revision_inventory_verified") is True
        and natural_audit.get("upstream_code_license_revision_inventory_verified") is True
        and natural_audit.get("ruler_license_revision_manifest_verified") is True
        and natural_audit.get("model_license_revision_manifest_verified") is True
        and natural_audit.get("safety_stress_terminal") is True
        and natural_audit.get("natural_safety_terminal") is True
        and natural_audit.get("benchmarks_terminal") == 5
        and natural_audit.get("minimum_protocol_examples_accounted_per_arm") == 45_289
    )
    safety_audit = p3_safety["audit"]
    safety_complete = (
        safety_audit.get("required_arms_terminal") is True
        and safety_audit.get("failure_accounting_complete") is True
        and safety_audit.get("input_pairing_verified") is True
        and safety_audit.get("source_implementations_verified") is True
        and safety_audit.get("coordinate_grid_verified") is True
        and safety_audit.get("record_revisions_verified") is True
        and safety_audit.get("terminal_measurement_schema_verified") is True
        and safety_audit.get("target_and_canary_pairing_verified") is True
        and safety_audit.get("protected_prefix_physical_budget_verified") is True
        and safety_audit.get("raw_artifact_digests_verified") is True
        and safety_audit.get("statistical_schema_verified") is True
        and safety_audit.get("examples_accounted_per_arm") == 1_200
        and safety_audit.get("families_terminal") == 4
        and safety_audit.get("contexts_terminal") == 3
    )
    natural_safety_audit = p3_natural_safety["audit"]
    natural_safety_complete = (
        natural_safety_audit.get("required_arms") == 2
        and natural_safety_audit.get("longsafety_generation_terminal") is True
        and natural_safety_audit.get("longsafety_input_pairing_verified") is True
        and natural_safety_audit.get("longsafety_expected_generations_per_arm") == 3_086
        and natural_safety_audit.get("longsafety_official_judge_status") == "blocked"
        and natural_safety_audit.get("longsafety_safety_scores_reported") is False
        and natural_safety_audit.get("longsafety_raw_evidence_verified") is True
        and natural_safety_audit.get("ifeval_official_terminal") is True
        and natural_safety_audit.get("ifeval_input_pairing_verified") is True
        and natural_safety_audit.get("ifeval_expected_prompts_per_arm") == 541
        and natural_safety_audit.get("ifeval_raw_evidence_verified") is True
        and natural_safety_audit.get("failure_accounting_complete") is True
        and natural_safety_audit.get("source_implementations_verified") is True
        and natural_safety_audit.get("raw_artifact_digests_verified") is True
        and natural_safety_audit.get("statistical_schema_verified") is True
        and natural_safety_audit.get("comparative_long_context_safety_claim_available") is False
    )
    ifeval_audit = p3_ifeval["audit"]
    ifeval_complete = (
        ifeval_audit.get("required_arms_terminal") is True
        and ifeval_audit.get("input_pairing_verified") is True
        and ifeval_audit.get("official_scoring_accounted") is True
        and ifeval_audit.get("source_implementations_verified") is True
        and ifeval_audit.get("generation_dependency_digests_verified") is True
        and ifeval_audit.get("generation_record_revisions_verified") is True
        and ifeval_audit.get("generation_terminal_measurement_schema_verified") is True
        and ifeval_audit.get("generation_seed_verified") is True
        and ifeval_audit.get("official_result_schema_verified") is True
        and ifeval_audit.get("expected_prompts_per_arm") == 541
    )
    longsafety_audit = p3_longsafety["audit"]
    longsafety_judged = (
        longsafety_audit.get("generation_arms_terminal") is True
        and longsafety_audit.get("input_pairing_verified") is True
        and longsafety_audit.get("generation_failure_accounting_complete") is True
        and longsafety_audit.get("source_implementations_verified") is True
        and longsafety_audit.get("dependency_digests_verified") is True
        and longsafety_audit.get("record_revisions_verified") is True
        and longsafety_audit.get("terminal_measurement_schema_verified") is True
        and longsafety_audit.get("generation_seed_verified") is True
        and longsafety_audit.get("expected_generations_total") == 6_172
        and longsafety_audit.get("official_judge_status") == "complete"
    )
    preflight_class = _classify_500k_preflight(p4_500k_context)
    reference_audit = p4_reference_systems["audit"]
    reference_counts = tuple(
        reference_audit.get(field) for field in ("complete_cells", "partial_cells", "failed_cells")
    )
    reference_accounted = (
        reference_audit.get("terminal_cells") == P4_EXPECTED_CELLS
        and reference_audit.get("repetition_seed_schedule_verified") is True
        and reference_audit.get("raw_latency_samples_and_derived_statistics_verified") is True
        and reference_audit.get("input_seed_base") == 9_071_400
        and all(type(value) is int and value >= 0 for value in reference_counts)
        and sum(reference_counts) == P4_EXPECTED_CELLS
    )
    reference_measured = reference_accounted and sum(reference_counts[:2]) > 0
    adaptive_audit = (
        p4_adaptive_systems.get("audit", {}) if isinstance(p4_adaptive_systems, dict) else {}
    )
    adaptive_counts = tuple(
        adaptive_audit.get(field) for field in ("complete_cells", "partial_cells", "failed_cells")
    )
    adaptive_counts_valid = all(type(value) is int and value >= 0 for value in adaptive_counts)
    adaptive_count_values = (
        tuple(cast(int, value) for value in adaptive_counts)
        if adaptive_counts_valid
        else (-1, -1, -1)
    )
    adaptive_accounted = (
        adaptive_audit.get("terminal_cells") == 432
        and adaptive_audit.get("fixed_calibrated_policy_pair_verified") is True
        and adaptive_audit.get("both_budgets_verified") is True
        and adaptive_audit.get("both_scales_verified") is True
        and adaptive_audit.get("outcome_independent_execution_verified") is True
        and adaptive_audit.get("adaptive_controller_measurement_verified") is True
        and adaptive_audit.get("physical_hot_budget_schema_verified") is True
        and adaptive_audit.get("raw_latency_samples_and_derived_statistics_verified") is True
        and adaptive_audit.get("tail_failure_accounting_complete") is True
        and adaptive_audit.get("input_seed_base") == 9_171_400
        and adaptive_counts_valid
        and sum(adaptive_count_values) == 432
    )
    adaptive_class = (
        "bounded-result"
        if adaptive_accounted and sum(adaptive_count_values[:2]) > 0
        else "negative-result"
        if adaptive_accounted
        else "unverified"
    )
    adaptive_production_audit = (
        p4_adaptive_production_systems.get("audit", {})
        if isinstance(p4_adaptive_production_systems, dict)
        else {}
    )
    adaptive_production_counts = tuple(
        adaptive_production_audit.get(field)
        for field in ("complete_cells", "partial_cells", "failed_cells")
    )
    adaptive_production_accounted = (
        adaptive_production_audit.get("terminal_cells") == 432
        and adaptive_production_audit.get("all_terminal_cells_verified") is True
        and adaptive_production_audit.get("adapter_spec_and_controller_schedule_verified")
        is True
        and adaptive_production_audit.get("fixed_calibrated_policy_pair_verified") is True
        and adaptive_production_audit.get("physical_hot_budget_schema_verified") is True
        and adaptive_production_audit.get("raw_latency_samples_and_derived_statistics_verified")
        is True
        and adaptive_production_audit.get("failure_accounting_complete") is True
        and adaptive_production_audit.get("dynamic_arrivals_or_continuous_admission_verified")
        is False
        and adaptive_production_audit.get("external_fused_runtime_verified") is False
        and adaptive_production_audit.get("kernel_aware_residency_layout_verified")
        is False
        and adaptive_production_audit.get("position_aware_recomputation_cost_verified")
        is False
        and adaptive_production_audit.get("fused_attention_kernel_cost_model_verified")
        is False
        and all(type(value) is int and value >= 0 for value in adaptive_production_counts)
        and sum(cast(int, value) for value in adaptive_production_counts) == 432
    )
    adaptive_production_class = (
        "bounded-result"
        if adaptive_production_accounted
        and adaptive_production_audit.get("complete_cells", 0) > 0
        else "negative-result"
        if adaptive_production_accounted
        else "unverified"
    )
    production_audit = p4_production_systems["audit"]
    production_counts = tuple(
        production_audit.get(field) for field in ("complete_cells", "partial_cells", "failed_cells")
    )
    production_seed_evidence = (
        production_audit.get("adapter_spec_digests_and_seed_schedule_verified") is True
        and production_audit.get("repetition_seed_schedule_verified") is True
        and production_audit.get("input_seed_base") == 9_071_400
    )
    production_accounted = (
        production_audit.get("terminal_cells") == P4_EXPECTED_CELLS
        and production_seed_evidence
        and production_audit.get("raw_latency_samples_and_derived_statistics_verified") is True
        and all(type(value) is int and value >= 0 for value in production_counts)
        and sum(production_counts) == P4_EXPECTED_CELLS
        and production_audit.get("tail_failure_accounting_complete") is True
        and production_audit.get("failure_provenance_verified") is True
    )
    production_full = (
        production_accounted
        and production_audit.get("complete_cells") == P4_EXPECTED_CELLS
        and production_audit.get("partial_cells") == 0
        and production_audit.get("failed_cells") == 0
        and production_audit.get("actual_concurrency_verified") is True
        and production_audit.get("all_required_metrics_verified") is True
        and production_audit.get("warmup_accounting_status_recorded") is True
        and production_audit.get("warmup_accounting_available_all_adapter_cells") is True
        and production_audit.get("whole_cell_timeout_contract_verified") is True
        and production_audit.get("allocator_hbm_metrics_verified") is True
        and production_audit.get("process_total_hbm_availability_accounted") is True
        and production_audit.get("backend_provenance_consistent") is True
        and production_audit.get("tail_failure_accounting_complete") is True
        and production_audit.get("all_paired_predictions_identical") is True
    )
    production_external_runtime_verified = (
        production_audit.get("external_fused_dynamic_runtime_verified") is True
    )
    production_verified_kernel_contract = (
        production_audit.get("kernel_aware_residency_layout_verified") is True
        and production_audit.get("position_aware_recomputation_cost_verified") is True
        and production_audit.get("fused_attention_kernel_cost_model_verified") is True
    )
    production_unverified_kernel_boundary = (
        production_audit.get("kernel_aware_residency_layout_verified") is False
        and production_audit.get("position_aware_recomputation_cost_verified") is False
        and production_audit.get("fused_attention_kernel_cost_model_verified") is False
    )
    production_class = (
        "success"
        if production_full
        and production_external_runtime_verified
        and production_verified_kernel_contract
        else "bounded-result"
        if production_accounted
        and production_audit.get("complete_cells", 0) > 0
        and production_unverified_kernel_boundary
        else "unverified"
    )
    result = {
        "p2_core": (
            "success" if core_passed else "negative-result" if core_complete else "unverified"
        ),
        "m5_one_token_pilot": "negative-result" if m5_complete else "unverified",
        "m3_offline_learned_risk_pilot": ("negative-result" if m3_complete else "unverified"),
        "p1_online_learned_lookahead": (
            "success"
            if learned_complete and learned_passed
            else "negative-result"
            if learned_complete
            else "unverified"
        ),
        "p2_causal": "success" if causal_passed else "bounded-result",
        "p3_ruler": "bounded-result" if p3_complete else "unverified",
        "p3_natural": "bounded-result" if natural_complete else "unverified",
        "p3_safety": "bounded-result" if safety_complete else "unverified",
        "p3_natural_safety": ("bounded-result" if natural_safety_complete else "unverified"),
        "p3_ifeval": "bounded-result" if ifeval_complete else "unverified",
        "p3_longsafety": "bounded-result" if longsafety_judged else "unverified",
        "p4_500k_context": preflight_class,
        "p4_reference_systems": "bounded-result" if reference_measured else "unverified",
        "p4_adaptive_systems": adaptive_class,
        "p4_adaptive_production_systems": adaptive_production_class,
        "p4_production_systems": production_class,
        "production_runtime_blocker": "unverified",
        "official_deepseek_v4": "unverified",
    }
    if isinstance(p3_cross_family, dict):
        cross_audit = p3_cross_family.get("audit", {})
        cross_terminal = (
            p3_cross_family.get("status") == "terminal"
            and cross_audit.get("terminal_arms") == 2
            and cross_audit.get("total_predictions") == 7_800
            and cross_audit.get("paired_examples") == 3_900
            and cross_audit.get("all_scores_recomputed_from_raw_response") is True
            and cross_audit.get("all_runtime_kvpress_bindings_verified") is True
            and cross_audit.get("all_dependency_digests_verified") is True
            and cross_audit.get("exact_input_pairing_verified") is True
            and cross_audit.get("exact_token_contract_verified") is True
            and cross_audit.get("failure_accounting_complete") is True
            and cross_audit.get("physical_kv_measurements_verified") is True
            and cross_audit.get("phi_specific_reselection") is False
            and cross_audit.get("outcome_dependent_execution") is False
        )
        cross_gate = p3_cross_family.get("transfer_gate", {})
        result["p3_cross_family"] = (
            "success"
            if cross_terminal and cross_gate.get("passed") is True
            else "negative-result"
            if cross_terminal and cross_gate.get("passed") is False
            else "unverified"
        )
    if isinstance(p3_natural_adaptive_quota, dict):
        adaptive_natural_audit = p3_natural_adaptive_quota.get("audit", {})
        adaptive_natural_terminal = (
            p3_natural_adaptive_quota.get("status") == "terminal"
            and adaptive_natural_audit.get("terminal_arms") == 2
            and adaptive_natural_audit.get("total_predictions") == 65_000
            and adaptive_natural_audit.get("paired_examples") == 32_500
            and adaptive_natural_audit.get("all_scores_recomputed_from_raw_response") is True
            and adaptive_natural_audit.get("all_runtime_kvpress_bindings_verified") is True
            and adaptive_natural_audit.get("all_dependency_digests_verified") is True
            and adaptive_natural_audit.get("exact_input_pairing_verified") is True
            and adaptive_natural_audit.get("quota_physical_audits_verified") is True
            and adaptive_natural_audit.get("same_global_token_budget_verified") is True
            and adaptive_natural_audit.get("causal_layer_order_verified") is True
            and adaptive_natural_audit.get("failure_accounting_complete") is True
            and adaptive_natural_audit.get("record_revision_provenance_verified") is True
            and adaptive_natural_audit.get("model_snapshot_digest_set_verified") is True
            and adaptive_natural_audit.get("operational_failure_vocabulary_verified") is True
            and adaptive_natural_audit.get("synthetic_controller_unchanged_transfer") is False
            and adaptive_natural_audit.get("outcome_dependent_execution") is False
        )
        adaptive_natural_gate = p3_natural_adaptive_quota.get("analysis", {}).get(
            "confirmation_gate", {}
        )
        result["p3_natural_adaptive_quota"] = (
            "success"
            if adaptive_natural_terminal and adaptive_natural_gate.get("passed") is True
            else "negative-result"
            if adaptive_natural_terminal and adaptive_natural_gate.get("passed") is False
            else "unverified"
        )
    if isinstance(p3_natural_adaptive_quota_scbench, dict):
        scbench_audit = p3_natural_adaptive_quota_scbench.get("audit", {})
        scbench_terminal = (
            p3_natural_adaptive_quota_scbench.get("status") == "terminal"
            and scbench_audit.get("terminal_arms") == 2
            and scbench_audit.get("total_predictions") == 20_572
            and scbench_audit.get("paired_turns") == 10_286
            and scbench_audit.get("all_raw_records_verified") is True
            and scbench_audit.get("all_scores_recomputed_from_raw_response") is True
            and scbench_audit.get("all_dependency_digests_verified") is True
            and scbench_audit.get("exact_input_pairing_verified") is True
            and scbench_audit.get("shared_context_cluster_pairing_verified") is True
            and scbench_audit.get("initial_prefill_quota_audits_verified") is True
            and scbench_audit.get("same_initial_global_token_budget_verified") is True
            and scbench_audit.get("failure_accounting_complete") is True
            and scbench_audit.get("operational_failure_vocabulary_verified") is True
            and scbench_audit.get("continuous_refresh_claim_available") is False
            and scbench_audit.get("outcome_dependent_execution") is False
        )
        scbench_gate = p3_natural_adaptive_quota_scbench.get("confirmation_gate", {})
        result["p3_natural_adaptive_quota_scbench"] = (
            "success"
            if scbench_terminal and scbench_gate.get("passed") is True
            else "negative-result"
            if scbench_terminal and scbench_gate.get("passed") is False
            else "unverified"
        )
    if isinstance(p3_natural_adaptive_quota_longbench_v2, dict):
        longbench_audit = p3_natural_adaptive_quota_longbench_v2.get("audit", {})
        longbench_terminal = (
            p3_natural_adaptive_quota_longbench_v2.get("status") == "terminal"
            and longbench_audit.get("terminal_arms") == 2
            and longbench_audit.get("total_predictions") == 1_006
            and longbench_audit.get("paired_examples") == 503
            and longbench_audit.get("all_raw_records_verified") is True
            and longbench_audit.get("all_scores_recomputed_from_raw_response") is True
            and longbench_audit.get("all_dependency_digests_verified") is True
            and longbench_audit.get("exact_input_pairing_verified") is True
            and longbench_audit.get("exact_token_id_pairing_verified") is True
            and longbench_audit.get("quota_physical_audits_verified") is True
            and longbench_audit.get("same_initial_global_token_budget_verified") is True
            and longbench_audit.get("failure_accounting_complete") is True
            and longbench_audit.get("operational_failure_vocabulary_verified") is True
            and longbench_audit.get("category_cells") == 6
            and longbench_audit.get("holm_family_size") == 6
            and longbench_audit.get("continuous_refresh_claim_available") is False
            and longbench_audit.get("outcome_dependent_execution") is False
        )
        longbench_gate = p3_natural_adaptive_quota_longbench_v2.get(
            "confirmation_gate", {}
        )
        result["p3_natural_adaptive_quota_longbench_v2"] = (
            "success"
            if longbench_terminal and longbench_gate.get("passed") is True
            else "negative-result"
            if longbench_terminal and longbench_gate.get("passed") is False
            else "unverified"
        )
    if isinstance(p3_natural_adaptive_quota_longmemeval, dict):
        result["p3_natural_adaptive_quota_longmemeval"] = "unverified"
    if isinstance(p3_natural_adaptive_quota_mrcr, dict):
        mrcr_audit = p3_natural_adaptive_quota_mrcr.get("audit", {})
        mrcr_terminal = (
            p3_natural_adaptive_quota_mrcr.get("status") == "terminal"
            and mrcr_audit.get("terminal_arms") == 2
            and mrcr_audit.get("total_predictions") == 3_000
            and mrcr_audit.get("paired_examples") == 1_500
            and mrcr_audit.get("all_raw_records_verified") is True
            and mrcr_audit.get("all_scores_recomputed_from_raw_response") is True
            and mrcr_audit.get("all_dependency_digests_verified") is True
            and mrcr_audit.get("exact_input_pairing_verified") is True
            and mrcr_audit.get("exact_token_id_pairing_verified") is True
            and mrcr_audit.get("quota_physical_audits_verified") is True
            and mrcr_audit.get("same_initial_global_token_budget_verified") is True
            and mrcr_audit.get("failure_accounting_complete") is True
            and mrcr_audit.get("operational_failure_vocabulary_verified") is True
            and mrcr_audit.get("needle_token_bin_cells") == 15
            and mrcr_audit.get("holm_family_size") == 15
            and mrcr_audit.get("continuous_refresh_claim_available") is False
            and mrcr_audit.get("outcome_dependent_execution") is False
        )
        mrcr_gate = p3_natural_adaptive_quota_mrcr.get("confirmation_gate", {})
        result["p3_natural_adaptive_quota_mrcr"] = (
            "success"
            if mrcr_terminal and mrcr_gate.get("passed") is True
            else "negative-result"
            if mrcr_terminal and mrcr_gate.get("passed") is False
            else "unverified"
        )
    if isinstance(p3_natural_adaptive_quota_suite, dict):
        suite_audit = p3_natural_adaptive_quota_suite.get("audit", {})
        suite_terminal = (
            p3_natural_adaptive_quota_suite.get("status") == "terminal"
            and suite_audit.get("terminal_components") == 4
            and suite_audit.get("predictions_per_arm") == 44_789
            and suite_audit.get("total_predictions") == 89_578
            and suite_audit.get("all_component_manifest_digests_verified") is True
            and suite_audit.get("all_component_summary_digests_recorded") is True
            and suite_audit.get("all_component_arm_cell_digests_verified") is True
            and suite_audit.get("all_component_raw_audits_verified") is True
            and suite_audit.get("all_component_score_recomputation_verified") is True
            and suite_audit.get("all_component_dependency_digests_verified") is True
            and suite_audit.get("all_component_runtime_bindings_verified") is True
            and suite_audit.get("all_component_failure_accounting_verified") is True
            and suite_audit.get("same_model_revision_verified") is True
            and suite_audit.get("same_arm_pair_verified") is True
            and suite_audit.get("same_initial_global_token_budget_verified") is True
            and suite_audit.get("no_cross_benchmark_score_pooling") is True
            and suite_audit.get("no_cross_benchmark_p_value_pooling") is True
            and suite_audit.get("longmemeval_official_judge_boundary_preserved") is True
            and suite_audit.get("outcome_dependent_benchmark_selection") is False
        )
        suite_gate = p3_natural_adaptive_quota_suite.get("suite_confirmation_gate", {})
        result["p3_natural_adaptive_quota_suite"] = (
            "success"
            if suite_terminal and suite_gate.get("passed") is True
            else "negative-result"
            if suite_terminal and suite_gate.get("passed") is False
            else "unverified"
        )
    if isinstance(p3_cross_family_adaptive_quota, dict):
        cross_adaptive_audit = p3_cross_family_adaptive_quota.get("audit", {})
        cross_adaptive_terminal = (
            p3_cross_family_adaptive_quota.get("status") == "terminal"
            and cross_adaptive_audit.get("terminal_arms") == 2
            and cross_adaptive_audit.get("total_predictions") == 7_800
            and cross_adaptive_audit.get("paired_examples") == 3_900
            and cross_adaptive_audit.get("all_raw_records_verified") is True
            and cross_adaptive_audit.get("all_scores_recomputed_from_raw_response") is True
            and cross_adaptive_audit.get("all_dependency_digests_verified") is True
            and cross_adaptive_audit.get("operational_failure_vocabulary_verified") is True
            and cross_adaptive_audit.get("exact_input_pairing_verified") is True
            and cross_adaptive_audit.get("quota_physical_audits_verified") is True
            and cross_adaptive_audit.get("same_global_token_budget_verified") is True
            and cross_adaptive_audit.get("phi_specific_reselection") is False
            and cross_adaptive_audit.get("pooled_with_qwen") is False
            and cross_adaptive_audit.get("outcome_dependent_execution") is False
        )
        cross_adaptive_gate = p3_cross_family_adaptive_quota.get("confirmation_gate", {})
        result["p3_cross_family_adaptive_quota"] = (
            "success"
            if cross_adaptive_terminal and cross_adaptive_gate.get("passed") is True
            else "negative-result"
            if cross_adaptive_terminal and cross_adaptive_gate.get("passed") is False
            else "unverified"
        )
    if isinstance(p3_cross_family_adaptive_quota_longbench_v2, dict):
        phi_longbench_audit = p3_cross_family_adaptive_quota_longbench_v2.get(
            "audit", {}
        )
        phi_longbench_terminal = (
            p3_cross_family_adaptive_quota_longbench_v2.get("status") == "terminal"
            and phi_longbench_audit.get("required_arms_terminal") is True
            and phi_longbench_audit.get("terminal_arms") == 2
            and phi_longbench_audit.get("predictions_per_arm") == 503
            and phi_longbench_audit.get("total_predictions") == 1_006
            and phi_longbench_audit.get("paired_examples") == 503
            and phi_longbench_audit.get("all_raw_records_verified") is True
            and phi_longbench_audit.get("all_scores_recomputed_from_raw_response") is True
            and phi_longbench_audit.get("all_dependency_digests_verified") is True
            and phi_longbench_audit.get("all_runtime_kvpress_bindings_verified") is True
            and phi_longbench_audit.get("exact_input_pairing_verified") is True
            and phi_longbench_audit.get("exact_token_id_pairing_verified") is True
            and phi_longbench_audit.get("quota_physical_audits_verified") is True
            and phi_longbench_audit.get("same_initial_global_token_budget_verified")
            is True
            and phi_longbench_audit.get("causal_layer_order_verified") is True
            and phi_longbench_audit.get("failure_accounting_complete") is True
            and phi_longbench_audit.get("operational_failure_vocabulary_verified")
            is True
            and phi_longbench_audit.get("record_revision_provenance_verified") is True
            and phi_longbench_audit.get("model_snapshot_digest_set_verified") is True
            and phi_longbench_audit.get("phi_specific_reselection") is False
            and phi_longbench_audit.get("qwen_selection_reused_without_phi_tuning")
            is True
            and phi_longbench_audit.get("category_cells") == 6
            and phi_longbench_audit.get("holm_family_size") == 6
            and phi_longbench_audit.get("paired_bootstrap_resamples") == 10_000
            and phi_longbench_audit.get("paired_bootstrap_seed") == 9_671_507
            and phi_longbench_audit.get("adaptive_allocation_scope")
            == "initial context prefill only"
            and phi_longbench_audit.get("continuous_refresh_claim_available") is False
            and phi_longbench_audit.get("secondary_slices_are_descriptive") is True
            and phi_longbench_audit.get("outcome_dependent_execution") is False
        )
        phi_longbench_gate = p3_cross_family_adaptive_quota_longbench_v2.get(
            "confirmation_gate", {}
        )
        result["p3_cross_family_adaptive_quota_longbench_v2"] = (
            "success"
            if phi_longbench_terminal and phi_longbench_gate.get("passed") is True
            else "negative-result"
            if phi_longbench_terminal and phi_longbench_gate.get("passed") is False
            else "unverified"
        )
    _require(set(result.values()).issubset(ALLOWED_CLASSES), "Unknown conclusion class.")
    return result


def _validate_required_classifications(
    classifications: dict[str, str], contract: Any
) -> None:
    _require(
        isinstance(contract, dict) and bool(contract),
        "Missing P5 required-classification contract.",
    )
    _require(
        set(contract) == set(classifications),
        "P5 required-classification coverage drifted.",
    )
    for name, allowed in contract.items():
        _require(
            isinstance(allowed, list)
            and bool(allowed)
            and all(value in ALLOWED_CLASSES for value in allowed),
            f"Invalid P5 required classifications for {name}.",
        )
        observed = classifications[name]
        _require(
            observed in allowed,
            f"P5 classification {name}={observed} is not terminally admissible; "
            f"expected one of {allowed}.",
        )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _flatten_json_row(row: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flattened[f"{key}.{child_key}"] = (
                    json.dumps(child_value, sort_keys=True, separators=(",", ":"))
                    if isinstance(child_value, (dict, list))
                    else child_value
                )
        elif isinstance(value, list):
            flattened[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            flattened[key] = value
    return flattened


def _field_union(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    _require(bool(fields), "Cannot write a table without fields.")
    return fields


def _write_interval_svg(
    path: Path,
    *,
    title: str,
    subtitle: str,
    x_label: str,
    rows: list[dict[str, Any]],
    source: dict[str, Any],
) -> None:
    """Write a deterministic, dependency-free forest plot with embedded provenance."""

    for row in rows:
        values = (row.get("lower"), row.get("value"), row.get("upper"))
        _require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in values
            ),
            "Figure interval contains a non-finite value.",
        )
        numeric_values = cast(tuple[float | int, float | int, float | int], values)
        _require(
            numeric_values[0] <= numeric_values[1] <= numeric_values[2],
            "Figure interval order drifted.",
        )
    width = 1_080
    left = 330
    right = 140
    top = 112
    row_height = 42
    plot_width = width - left - right
    height = max(250, top + max(1, len(rows)) * row_height + 92)
    observed = [0.0]
    for row in rows:
        observed.extend((float(row["lower"]), float(row["upper"])))
    minimum = min(observed)
    maximum = max(observed)
    span = maximum - minimum
    padding = max(0.5, span * 0.12)
    x_min = minimum - padding
    x_max = maximum + padding
    if x_min == x_max:
        x_min, x_max = -1.0, 1.0

    def x_position(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    metadata = {
        "schema_version": 1,
        "source": source,
        "rows_sha256": _canonical_digest(rows),
        "rows": rows,
    }
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(subtitle)}</desc>',
        f"<metadata>{html.escape(json.dumps(metadata, sort_keys=True))}</metadata>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        (
            f'<text x="32" y="38" font-family="sans-serif" font-size="22" '
            f'font-weight="700" fill="#172033">{html.escape(title)}</text>'
        ),
        (
            f'<text x="32" y="66" font-family="sans-serif" font-size="13" '
            f'fill="#516071">{html.escape(subtitle)}</text>'
        ),
    ]
    axis_y = height - 58
    zero_x = x_position(0.0)
    lines.append(
        f'<line x1="{zero_x:.2f}" y1="88" x2="{zero_x:.2f}" y2="{axis_y}" '
        'stroke="#8b96a5" stroke-width="1.5" stroke-dasharray="4 4"/>'
    )
    for index in range(5):
        value = x_min + (x_max - x_min) * index / 4
        x = x_position(value)
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{axis_y}" x2="{x:.2f}" y2="{axis_y + 6}" stroke="#566273"/>',
                (
                    f'<text x="{x:.2f}" y="{axis_y + 24}" text-anchor="middle" '
                    f'font-family="monospace" font-size="11" fill="#516071">{value:.2f}</text>'
                ),
            ]
        )
    if not rows:
        lines.append(
            '<text x="540" y="145" text-anchor="middle" font-family="sans-serif" '
            'font-size="16" fill="#8a3b31">No paired measured cells; terminal failures are retained.</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * row_height
        lower = x_position(float(row["lower"]))
        value = x_position(float(row["value"]))
        upper = x_position(float(row["upper"]))
        color = str(row.get("color", "#176b87"))
        label = html.escape(str(row["label"]))
        lines.extend(
            [
                f'<text x="{left - 18}" y="{y + 5}" text-anchor="end" font-family="sans-serif" font-size="13" fill="#253247">{label}</text>',
                f'<line x1="{lower:.2f}" y1="{y}" x2="{upper:.2f}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<line x1="{lower:.2f}" y1="{y - 6}" x2="{lower:.2f}" y2="{y + 6}" stroke="{color}"/>',
                f'<line x1="{upper:.2f}" y1="{y - 6}" x2="{upper:.2f}" y2="{y + 6}" stroke="{color}"/>',
                f'<circle cx="{value:.2f}" cy="{y}" r="5" fill="{color}"/>',
                (
                    f'<text x="{width - right + 12}" y="{y + 5}" font-family="monospace" '
                    f'font-size="11" fill="#253247">{float(row["value"]):+.2f} '
                    f"[{float(row['lower']):+.2f}, {float(row['upper']):+.2f}]</text>"
                ),
            ]
        )
    lines.append(
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 10}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="12" fill="#253247">{html.escape(x_label)}</text>'
    )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_p2_causal_figure(path: Path, payload: dict[str, Any]) -> None:
    cells = payload["paired_statistics"]["adaptive_quota_with_pins"]["cells"]
    _require(len(cells) == 4, "P2 causal figure requires all four primary cells.")
    rows = []
    for cell in sorted(cells, key=lambda row: (row["scale"], row["budget"])):
        interval = cell["four_cell_corrected_bootstrap"]["confidence_interval"]
        rows.append(
            {
                "label": f"{cell['scale']} · {cell['budget']}",
                "value": float(cell["mean_difference_percentage_points"]),
                "lower": float(interval[0]) * 100.0,
                "upper": float(interval[1]) * 100.0,
                "color": "#16734a" if interval[0] > 0.0 else "#a84b37",
            }
        )
    _write_interval_svg(
        path,
        title="Causal adaptive-quota effect at matched hot memory",
        subtitle="calibrated+pins minus fixed+pins; 98.75% seed-cluster bootstrap intervals",
        x_label="paired conversation accuracy difference (percentage points)",
        rows=rows,
        source={
            "experiment_id": payload["experiment_id"],
            "raw_matrix_sha256": payload["raw_matrix"]["sha256"],
            "contrast": "adaptive_quota_with_pins",
        },
    )


def _write_p4_tradeoff_figure(path: Path, payload: dict[str, Any]) -> None:
    metric_labels = {
        "ttft_p95_ms": "TTFT p95",
        "throughput_tokens_per_second": "Throughput",
        "peak_allocated_bytes": "HBM peak",
    }
    grouped: dict[tuple[int, str], list[float]] = {}
    measured_cells = [
        *payload["complete_cell_statistics"],
        *payload["partial_cell_statistics"],
    ]
    for cell in measured_cells:
        context = int(cell["cell"]["context"])
        for metric in metric_labels:
            ratio = cell.get("metrics", {}).get(metric, {}).get("mean_ratio_tiered_over_resident")
            if (
                isinstance(ratio, (int, float))
                and not isinstance(ratio, bool)
                and math.isfinite(ratio)
            ):
                grouped.setdefault((context, metric), []).append((float(ratio) - 1.0) * 100.0)
    rows = []
    for (context, metric), values in sorted(grouped.items()):
        mean = sum(values) / len(values)
        rows.append(
            {
                "label": f"{context // 1024}K · {metric_labels[metric]} (n={len(values)})",
                "value": mean,
                "lower": min(values),
                "upper": max(values),
                "color": "#7047a3" if metric == "throughput_tokens_per_second" else "#176b87",
            }
        )
    audit = payload["audit"]
    _write_interval_svg(
        path,
        title="Production adapter trade-offs across measured cells",
        subtitle=(
            f"tiered relative to resident; mean and range; terminal cells: {audit['terminal_cells']}, "
            f"complete: {audit['complete_cells']}, partial: {audit['partial_cells']}, "
            f"failed: {audit['failed_cells']}"
        ),
        x_label="tiered over resident change (%)",
        rows=rows,
        source={
            "experiment_id": payload["experiment_id"],
            "raw_matrix_sha256": payload["raw_matrix"]["sha256"],
            "metrics": list(metric_labels),
        },
    )


def _p2_quality_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "corrected_positive_families_by_scale": json.dumps(
                row["corrected_positive_families_by_scale"], sort_keys=True
            ),
        }
        for row in payload["quality_gate"]
    ]


def _p2_inference_resolution_rows(
    core: dict[str, Any], causal: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for stage, payload in (("p2-core", core), ("p2-causal", causal)):
        audit = payload["audit"]
        clusters = audit["independent_seed_clusters_per_cell"]
        rows.append(
            {
                "stage": stage,
                "independent_seed_clusters_per_cell": clusters,
                "exact_sign_flip_assignments": 1 << clusters,
                "minimum_attainable_two_sided_seed_p": audit["minimum_attainable_two_sided_seed_p"],
                "exact_seed_randomization_verified": audit["exact_seed_randomization_verified"],
                "p_value_used_as_success_gate": audit["seed_p_values_used_as_success_gate"],
                "interpretation": (
                    "seed-level exact p-values are resolution-limited descriptive evidence; "
                    "within-seed examples do not add independent trained-model clusters"
                ),
            }
        )
    return rows


def _p2_core_effect_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row({"comparison": comparison, **row})
        for comparison, statistics in payload["paired_statistics"].items()
        for row in statistics["pooled_by_scale"]
    ]


def _p2_core_family_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row({"comparison": comparison, **row})
        for comparison, statistics in payload["paired_statistics"].items()
        for row in statistics["by_scale_family"]
    ]


def _p2_core_seed_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row({"comparison": comparison, **row})
        for comparison, statistics in payload["paired_statistics"].items()
        for row in statistics["by_seed"]
    ]


def _p2_core_seed_variance_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row({"comparison": comparison, **row})
        for comparison, statistics in payload["paired_statistics"].items()
        for row in statistics["seed_variance"]
    ]


def _p2_core_worst_slice_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for comparison, statistics in payload["paired_statistics"].items():
        rows.append(
            _flatten_json_row(
                {"comparison": comparison, "scope": "global", **statistics["worst_slice"]}
            )
        )
        rows.extend(
            _flatten_json_row({"comparison": comparison, "scope": "budget-scale", **row})
            for row in statistics["worst_slice_by_budget_scale"]
        )
    return rows


def _causal_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload["primary_causal_gate"]["cells"])


def _causal_contrast_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row(
            {
                "contrast": contrast,
                "candidate": statistics["candidate"],
                "comparator": statistics["comparator"],
                **row,
            }
        )
        for contrast, statistics in payload["paired_statistics"].items()
        for row in statistics["cells"]
    ]


def _causal_family_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row(
            {
                "contrast": contrast,
                "candidate": statistics["candidate"],
                "comparator": statistics["comparator"],
                **row,
            }
        )
        for contrast, statistics in payload["paired_statistics"].items()
        for row in statistics["by_family_with_holm_bonferroni"]
    ]


def _causal_seed_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row(
            {
                "contrast": contrast,
                "candidate": statistics["candidate"],
                "comparator": statistics["comparator"],
                **row,
            }
        )
        for contrast, statistics in payload["paired_statistics"].items()
        for row in statistics["by_seed"]
    ]


def _causal_seed_variance_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row(
            {
                "contrast": contrast,
                "candidate": statistics["candidate"],
                "comparator": statistics["comparator"],
                **row,
            }
        )
        for contrast, statistics in payload["paired_statistics"].items()
        for row in statistics["seed_variance"]
    ]


def _causal_worst_slice_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contrast, statistics in payload["paired_statistics"].items():
        identity = {
            "contrast": contrast,
            "candidate": statistics["candidate"],
            "comparator": statistics["comparator"],
        }
        rows.append(_flatten_json_row({**identity, "scope": "global", **statistics["worst_slice"]}))
        rows.extend(
            _flatten_json_row({**identity, "scope": "budget-scale", **row})
            for row in statistics["worst_slice_by_budget_scale"]
        )
    return rows


def _causal_physical_memory_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    physical = payload["physical_hot_memory"]
    return (
        [_flatten_json_row({"scope": "seed-match", **row}) for row in physical["by_seed"]]
        + [_flatten_json_row({"scope": "aggregate-match", **row}) for row in physical["aggregate"]]
        + [
            _flatten_json_row({"scope": "all-physical-arms", **row})
            for row in physical["all_physical_arms_by_seed"]
        ]
    )


def _causal_oracle_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    oracle = payload["offline_oracle_upper_bound"]
    return [
        _flatten_json_row(
            {
                "role": oracle["inference_role"],
                "selection_unit": oracle["selection_unit"],
                "used_for_primary_gate": oracle["used_for_primary_gate"],
                **row,
            }
        )
        for row in oracle["cells"]
    ]


def _learned_lookahead_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    systems = {
        (row["scale"], row["budget"]): row for row in payload["primary_gate"]["system_cells"]
    }
    rows = []
    for cell in payload["primary_gate"]["cells"]:
        system = systems[(cell["scale"], cell["budget"])]
        rows.append(
            {
                **cell,
                "seed_cluster_bootstrap_ci": json.dumps(
                    cell["seed_cluster_bootstrap_ci"], separators=(",", ":")
                ),
                **{
                    name: value
                    for name, value in system.items()
                    if name not in {"scale", "budget", "passed"}
                },
                "quality_passed": cell["passed"],
                "system_passed": system["passed"],
            }
        )
    return rows


def _p3_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "length_tokens": row["length_tokens"],
            "arm": row["arm"],
            "compression_ratio": row["compression_ratio"],
            "rows": row["rows"],
            "accuracy": row["accuracy"],
            "elapsed_seconds": row["elapsed_seconds"],
            "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
            "peak_cuda_reserved_bytes": row["peak_cuda_reserved_bytes"],
        }
        for row in payload["cell_summary"]
    ]


def _p3_cross_family_task_length_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [_flatten_json_row(row) for row in payload["statistics"]["by_task_length"]]


def _p3_cross_family_length_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row(row)
        for row in payload["statistics"]["by_length_with_exact_task_cluster_inference"]
    ]


def _p3_natural_adaptive_task_length_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [_flatten_json_row(row) for row in payload["analysis"]["by_task_length"]]


def _p3_natural_adaptive_length_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [_flatten_json_row(row) for row in payload["analysis"]["by_length"]]


def _p3_natural_adaptive_layer_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row(row)
        for row in payload["analysis"]["quota_audit"]["adaptive_per_layer_distributions"]
    ]


def _p3_adaptive_scbench_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = payload["analysis"]
    rows = [{"scope": "overall", **_flatten_json_row(analysis["overall"])}]
    rows.extend(
        {"scope": "mode", **_flatten_json_row(row)} for row in analysis["by_mode"]
    )
    rows.append(
        {
            "scope": "initial-prefill-physical",
            **_flatten_json_row(analysis["initial_prefill_physical"]),
        }
    )
    for arm, row in analysis["arms"].items():
        rows.append({"scope": "arm", "arm": arm, **_flatten_json_row(row)})
    return rows


def _p3_adaptive_scbench_mode_task_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [_flatten_json_row(row) for row in payload["analysis"]["by_mode_task"]]


def _p3_adaptive_longbench_summary_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    analysis = payload["analysis"]
    rows = [{"scope": "overall", **_flatten_json_row(analysis["overall"])}]
    rows.append(
        {
            "scope": "initial-prefill-physical",
            **_flatten_json_row(analysis["initial_prefill_physical"]),
        }
    )
    for arm, row in analysis["arms"].items():
        rows.append({"scope": "arm", "arm": arm, **_flatten_json_row(row)})
    return rows


def _p3_adaptive_longbench_category_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [_flatten_json_row(row) for row in payload["analysis"]["by_category"]]


def _p3_adaptive_longbench_slice_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"field": field, **_flatten_json_row(row)}
        for field, rows in payload["analysis"]["descriptive_slices"].items()
        for row in rows
    ]


def _p3_adaptive_longmemeval_summary_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    analysis = payload["analysis"]
    rows = [
        {"scope": "quality", **_flatten_json_row(analysis["quality"])},
        {
            "scope": "response-generation",
            **_flatten_json_row(analysis["response_generation"]),
        },
        {
            "scope": "initial-prefill-physical",
            **_flatten_json_row(analysis["initial_prefill_physical"]),
        },
    ]
    rows.extend(
        {"scope": "arm-generation-audit", "arm": arm, **_flatten_json_row(audit)}
        for arm, audit in payload["generation_audits"].items()
    )
    rows.extend(
        {"scope": "arm-measurements", "arm": arm, **_flatten_json_row(measurements)}
        for arm, measurements in analysis["measurements_by_arm"].items()
    )
    return rows


def _p3_adaptive_longmemeval_question_type_rows(
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"question_type": question_type, "arm": arm, **counts}
        for question_type, by_arm in payload["analysis"]["by_question_type"].items()
        for arm, counts in by_arm.items()
    ]


def _p3_adaptive_mrcr_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    analysis = payload["analysis"]
    rows = [{"scope": "overall", **_flatten_json_row(analysis["overall"])}]
    rows.append(
        {
            "scope": "initial-prefill-physical",
            **_flatten_json_row(analysis["initial_prefill_physical"]),
        }
    )
    for arm, row in analysis["arms"].items():
        rows.append({"scope": "arm", "arm": arm, **_flatten_json_row(row)})
    return rows


def _p3_adaptive_mrcr_cell_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _flatten_json_row(row)
        for row in payload["analysis"]["by_needle_token_bin"]
    ]


def _p3_adaptive_suite_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [_flatten_json_row(row) for row in payload["components"]]


def _p3_safety_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"arm": arm, **row}
        for arm, arm_payload in payload["arms"].items()
        for row in arm_payload["slices"]
    ]


def _p3_natural_arm_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, benchmark_payload in payload["benchmarks"].items():
        for arm, arm_payload in benchmark_payload["required_arms"].items():
            scored = arm_payload["measurements"]["scored_only"]
            rows.append(
                {
                    "benchmark": benchmark,
                    "arm": arm,
                    "status": "complete",
                    "expected_examples": arm_payload["expected_examples"],
                    "scored_examples": arm_payload["scored_examples"],
                    "failed_examples": arm_payload["failed_examples"],
                    "failure_rate": arm_payload["failure_rate"],
                    "failures_by_type": json.dumps(
                        arm_payload["failures_by_type"], sort_keys=True, separators=(",", ":")
                    ),
                    "mean_score_over_scored": arm_payload["mean_score_over_scored"],
                    "mean_score_over_all_expected_failures_zero": arm_payload[
                        "mean_score_over_all_expected_failures_zero"
                    ],
                    "scored_latency_ms_mean": (
                        scored["latency_ms"]["mean"] if scored["latency_ms"] else ""
                    ),
                    "scored_latency_ms_p95": (
                        scored["latency_ms"]["p95"] if scored["latency_ms"] else ""
                    ),
                    "scored_latency_ms_p99": (
                        scored["latency_ms"]["p99"] if scored["latency_ms"] else ""
                    ),
                    "scored_peak_hbm_bytes_mean": (
                        scored["peak_hbm_bytes"]["mean"] if scored["peak_hbm_bytes"] else ""
                    ),
                    "scored_peak_hbm_bytes_p95": (
                        scored["peak_hbm_bytes"]["p95"] if scored["peak_hbm_bytes"] else ""
                    ),
                    "scored_hot_resident_bytes_mean": (
                        scored["hot_resident_bytes"]["mean"] if scored["hot_resident_bytes"] else ""
                    ),
                    "scored_hot_resident_bytes_p95": (
                        scored["hot_resident_bytes"]["p95"] if scored["hot_resident_bytes"] else ""
                    ),
                }
            )
        for arm, disposition in benchmark_payload["conditional_arms"].items():
            rows.append(
                {
                    "benchmark": benchmark,
                    "arm": arm,
                    "status": disposition["status"],
                }
            )
    return rows


def _p3_natural_contrast_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for benchmark, benchmark_payload in payload["benchmarks"].items():
        quality = benchmark_payload["paired_quality_contrast"]
        measurements = benchmark_payload["paired_measurement_contrasts"]
        rows.append(
            {
                "benchmark": benchmark,
                **{
                    key: quality[key]
                    for key in (
                        "candidate",
                        "comparator",
                        "paired_examples",
                        "paired_clusters",
                        "cluster_unit",
                        "jointly_scored_examples",
                        "mean_difference",
                        "mean_difference_percentage_points",
                        "two_sided_bootstrap_p",
                        "cluster_mean_sample_standard_deviation",
                        "bootstrap_resamples",
                        "confidence_level",
                        "bootstrap_seed",
                    )
                },
                "paired_bootstrap_95_ci": json.dumps(
                    quality["paired_bootstrap_95_ci"], separators=(",", ":")
                ),
                "paired_bootstrap_95_ci_percentage_points": json.dumps(
                    quality["paired_bootstrap_95_ci_percentage_points"],
                    separators=(",", ":"),
                ),
                "failure_pairing": json.dumps(
                    quality["failure_pairing"], sort_keys=True, separators=(",", ":")
                ),
                "latency_mean_paired_difference": measurements["latency_ms"][
                    "mean_paired_difference"
                ],
                "latency_ratio_of_means": measurements["latency_ms"]["ratio_of_means"],
                "peak_hbm_mean_paired_difference": measurements["peak_hbm_bytes"][
                    "mean_paired_difference"
                ],
                "peak_hbm_ratio_of_means": measurements["peak_hbm_bytes"]["ratio_of_means"],
                "hot_resident_mean_paired_difference": measurements["hot_resident_bytes"][
                    "mean_paired_difference"
                ],
                "hot_resident_ratio_of_means": measurements["hot_resident_bytes"]["ratio_of_means"],
            }
        )
    return rows


def _write_p3_natural_figure(path: Path, payload: dict[str, Any]) -> None:
    rows = []
    summaries: dict[str, str] = {}
    for benchmark, benchmark_payload in payload["benchmarks"].items():
        quality = benchmark_payload["paired_quality_contrast"]
        interval = quality["paired_bootstrap_95_ci_percentage_points"]
        rows.append(
            {
                "label": benchmark,
                "value": quality["mean_difference_percentage_points"],
                "lower": interval[0],
                "upper": interval[1],
                "color": "#16734a" if interval[0] > 0.0 else "#a84b37",
            }
        )
        summaries[benchmark] = benchmark_payload["summary"]["sha256"]
    _write_interval_svg(
        path,
        title="Natural long-context quality against the preselected fixed baseline",
        subtitle=(
            "best of four frozen Qwen3-1.7B RULER candidates at 50% KV, transferred "
            "unchanged; fixed minus native; failures score zero"
        ),
        x_label="paired benchmark score difference (percentage points)",
        rows=rows,
        source={
            "experiment_id": payload["experiment_id"],
            "experiment_manifest_sha256": payload["experiment_manifest"]["sha256"],
            "benchmark_summary_sha256": summaries,
        },
    )


def _metric_mean(cell: dict[str, Any], metric: str, policy: str) -> Any:
    value = cell.get("metrics", {}).get(metric, {}).get(policy)
    return "" if value is None else value["mean"]


def _p4_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in [
        *payload["complete_cell_statistics"],
        *payload["partial_cell_statistics"],
    ]:
        coordinates = cell["cell"]
        row = {
            **coordinates,
            "active_requests": coordinates.get(
                "active_requests", coordinates.get("concurrency", "")
            ),
            "concurrency": coordinates.get("concurrency", ""),
            "status": cell["status"],
        }
        row.update(
            {
                "paired_repetitions": cell["paired_repetitions"],
                "cell_timeout_seconds": cell["cell_timeout_seconds"],
                "warmup_accounting_available": cell["warmup_accounting_available"],
                "warmup_repetitions_attempted": cell["warmup_repetitions_attempted"],
                "warmup_paired_repetitions_completed": cell["warmup_paired_repetitions_completed"],
                "warmup_failures": json.dumps(
                    cell["warmup_failures"], sort_keys=True, separators=(",", ":")
                ),
                "resident_ttft_p95_ms_mean": _metric_mean(cell, "ttft_p95_ms", "resident"),
                "tiered_ttft_p95_ms_mean": _metric_mean(cell, "ttft_p95_ms", "tiered"),
                "resident_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "resident"
                ),
                "tiered_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "tiered"
                ),
                "resident_peak_hbm_mean": _metric_mean(cell, "peak_allocated_bytes", "resident"),
                "tiered_peak_hbm_mean": _metric_mean(cell, "peak_allocated_bytes", "tiered"),
                "failure": (
                    ""
                    if cell["status"] == "complete"
                    else json.dumps(cell["policy_status"], sort_keys=True, separators=(",", ":"))
                ),
            }
        )
        rows.append(row)
    for failure in payload["failure_table"]:
        rows.append(
            {
                **failure["cell"],
                "active_requests": failure["cell"].get(
                    "active_requests", failure["cell"].get("concurrency", "")
                ),
                "concurrency": failure["cell"].get("concurrency", ""),
                "status": "failed",
                "paired_repetitions": 0,
                "cell_timeout_seconds": failure["cell_timeout_seconds"],
                "warmup_accounting_available": failure["warmup_accounting_available"],
                "warmup_repetitions_attempted": failure["warmup_repetitions_attempted"],
                "warmup_paired_repetitions_completed": failure[
                    "warmup_paired_repetitions_completed"
                ],
                "warmup_failures": json.dumps(
                    failure["warmup_failures"], sort_keys=True, separators=(",", ":")
                ),
                "failure": json.dumps(failure.get("policy_status"), sort_keys=True),
            }
        )
    return rows


def _p4_metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in [
        *payload["complete_cell_statistics"],
        *payload["partial_cell_statistics"],
    ]:
        coordinates = cell["cell"]
        for metric, metric_payload in sorted(cell["metrics"].items()):
            paired = metric_payload.get("tiered_minus_resident")
            for policy in ("resident", "tiered"):
                distribution_payload = metric_payload.get(policy)
                if distribution_payload is None:
                    continue
                rows.append(
                    {
                        **coordinates,
                        "active_requests": coordinates.get(
                            "active_requests", coordinates.get("concurrency", "")
                        ),
                        "concurrency": coordinates.get("concurrency", ""),
                        "status": cell["status"],
                        "metric": metric,
                        "policy": policy,
                        **distribution_payload,
                        "paired_observations": metric_payload["paired_observations"],
                        "mean_ratio_tiered_over_resident": metric_payload.get(
                            "mean_ratio_tiered_over_resident"
                        ),
                        "paired_tiered_minus_resident": (
                            json.dumps(paired, sort_keys=True, separators=(",", ":"))
                            if paired is not None
                            else ""
                        ),
                    }
                )
    return rows


def _p4_adaptive_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in [
        *payload["complete_cell_statistics"],
        *payload["partial_cell_statistics"],
    ]:
        coordinates = cell["cell"]
        rows.append(
            {
                **coordinates,
                "status": cell["status"],
                "paired_repetitions": cell["paired_repetitions"],
                "cell_timeout_seconds": cell["cell_timeout_seconds"],
                "warmup_repetitions_attempted": cell["warmup_repetitions_attempted"],
                "warmup_paired_repetitions_completed": cell["warmup_paired_repetitions_completed"],
                "warmup_failures": json.dumps(
                    cell["warmup_failures"], sort_keys=True, separators=(",", ":")
                ),
                "fixed_ttft_p95_ms_mean": _metric_mean(cell, "ttft_p95_ms", "fixed+pins"),
                "calibrated_ttft_p95_ms_mean": _metric_mean(cell, "ttft_p95_ms", "calibrated+pins"),
                "fixed_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "fixed+pins"
                ),
                "calibrated_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "calibrated+pins"
                ),
                "fixed_peak_hbm_mean": _metric_mean(cell, "peak_allocated_bytes", "fixed+pins"),
                "calibrated_peak_hbm_mean": _metric_mean(
                    cell, "peak_allocated_bytes", "calibrated+pins"
                ),
                "fixed_controller_time_ns_mean": _metric_mean(
                    cell, "controller_time_ns", "fixed+pins"
                ),
                "calibrated_controller_time_ns_mean": _metric_mean(
                    cell, "controller_time_ns", "calibrated+pins"
                ),
                "prediction_digest_mismatches": cell["prediction_digest_mismatches"],
                "failure": (
                    ""
                    if cell["status"] == "complete"
                    else json.dumps(
                        cell.get("policy_status", {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            }
        )
    for failure in payload["failure_table"]:
        rows.append(
            {
                **failure["cell"],
                "status": "failed",
                "paired_repetitions": 0,
                "cell_timeout_seconds": failure["cell_timeout_seconds"],
                "warmup_repetitions_attempted": failure["warmup_repetitions_attempted"],
                "warmup_paired_repetitions_completed": failure[
                    "warmup_paired_repetitions_completed"
                ],
                "warmup_failures": json.dumps(
                    failure["warmup_failures"], sort_keys=True, separators=(",", ":")
                ),
                "failure": json.dumps(
                    failure.get("policy_status"), sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return rows


def _p4_adaptive_metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in [
        *payload["complete_cell_statistics"],
        *payload["partial_cell_statistics"],
    ]:
        for metric, metric_payload in sorted(cell["metrics"].items()):
            paired = metric_payload.get("calibrated_minus_fixed")
            for policy in ("fixed+pins", "calibrated+pins"):
                distribution_payload = metric_payload.get(policy)
                if distribution_payload is None:
                    continue
                rows.append(
                    {
                        **cell["cell"],
                        "status": cell["status"],
                        "metric": metric,
                        "policy": policy,
                        **distribution_payload,
                        "paired_observations": metric_payload["paired_observations"],
                        "mean_ratio_calibrated_over_fixed": metric_payload.get(
                            "mean_ratio_calibrated_over_fixed"
                        ),
                        "paired_calibrated_minus_fixed": (
                            json.dumps(paired, sort_keys=True, separators=(",", ":"))
                            if paired is not None
                            else ""
                        ),
                    }
                )
    return rows


def _p4_adaptive_production_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in payload["cells"]:
        rows.append(
            {
                **cell["cell"],
                "status": cell["status"],
                "paired_repetitions": cell["paired_repetitions"],
                "cell_timeout_seconds": cell["cell_timeout_seconds"],
                "warmup_accounting_available": cell["warmup_accounting_available"],
                "warmup_repetitions_attempted": cell["warmup_repetitions_attempted"],
                "warmup_paired_repetitions_completed": cell[
                    "warmup_paired_repetitions_completed"
                ],
                "warmup_failures": json.dumps(
                    cell["warmup_failures"], sort_keys=True, separators=(",", ":")
                ),
                "fixed_ttft_p95_ms_mean": _metric_mean(
                    cell, "ttft_p95_ms", "fixed+pins"
                ),
                "calibrated_ttft_p95_ms_mean": _metric_mean(
                    cell, "ttft_p95_ms", "calibrated+pins"
                ),
                "fixed_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "fixed+pins"
                ),
                "calibrated_throughput_mean": _metric_mean(
                    cell, "throughput_tokens_per_second", "calibrated+pins"
                ),
                "fixed_peak_hbm_mean": _metric_mean(
                    cell, "peak_allocated_bytes", "fixed+pins"
                ),
                "calibrated_peak_hbm_mean": _metric_mean(
                    cell, "peak_allocated_bytes", "calibrated+pins"
                ),
                "fixed_controller_time_ns_mean": _metric_mean(
                    cell, "controller_time_ns", "fixed+pins"
                ),
                "calibrated_controller_time_ns_mean": _metric_mean(
                    cell, "controller_time_ns", "calibrated+pins"
                ),
                "failure": (
                    ""
                    if cell["status"] == "complete"
                    else json.dumps(
                        cell["policy_status"], sort_keys=True, separators=(",", ":")
                    )
                ),
            }
        )
    for failure in payload["failure_table"]:
        rows.append(
            {
                **failure["cell"],
                "status": "failed",
                "paired_repetitions": 0,
                "cell_timeout_seconds": failure["cell_timeout_seconds"],
                "warmup_accounting_available": False,
                "warmup_repetitions_attempted": None,
                "warmup_paired_repetitions_completed": None,
                "warmup_failures": "[]",
                "failure": json.dumps(
                    failure["policy_status"], sort_keys=True, separators=(",", ":")
                ),
            }
        )
    return rows


def _p4_adaptive_production_metric_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in payload["cells"]:
        for metric, metric_payload in sorted(cell["metrics"].items()):
            paired = metric_payload.get("calibrated_minus_fixed")
            for policy in ("fixed+pins", "calibrated+pins"):
                distribution_payload = metric_payload.get(policy)
                if distribution_payload is None:
                    continue
                rows.append(
                    {
                        **cell["cell"],
                        "status": cell["status"],
                        "metric": metric,
                        "policy": policy,
                        **distribution_payload,
                        "paired_repetitions": cell["paired_repetitions"],
                        "paired_calibrated_minus_fixed": (
                            json.dumps(paired, sort_keys=True, separators=(",", ":"))
                            if paired is not None
                            else ""
                        ),
                    }
                )
    return rows


def _p4_500k_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "scale": cell["scale"],
            "policy": policy,
            "context_tokens": payload["audit"]["context_tokens"],
            "generation_tokens": payload["audit"]["generation_tokens"],
            "cell_timeout_seconds": cell["cell_timeout_seconds"],
            "status": attempt["status"],
            "prediction_digest": attempt.get("prediction_digest"),
            "peak_allocated_bytes": attempt.get("peak_allocated_bytes"),
            "pinned_host_bytes": attempt.get("pinned_host_bytes"),
            "error_type": attempt.get("error_type"),
            "error": attempt.get("error"),
        }
        for cell in payload["cells"]
        for policy, attempt in cell["policy_attempts"].items()
    ]


def _report(
    *,
    classifications: dict[str, str],
    traceability_rows: list[dict[str, Any]],
    p2_core: dict[str, Any],
    p2_core_confirmatory: dict[str, Any],
    m5_one_token_pilot: dict[str, Any],
    m3_offline_learned_risk_pilot: dict[str, Any],
    p1_online_learned_lookahead: dict[str, Any],
    p2_causal: dict[str, Any],
    p2_causal_confirmatory: dict[str, Any],
    p3_ruler: dict[str, Any],
    p3_cross_family: dict[str, Any],
    p3_cross_family_adaptive_quota: dict[str, Any],
    p3_cross_family_adaptive_quota_longbench_v2: dict[str, Any],
    p3_natural_adaptive_quota: dict[str, Any],
    p3_natural_adaptive_quota_scbench: dict[str, Any],
    p3_natural_adaptive_quota_longbench_v2: dict[str, Any],
    p3_natural_adaptive_quota_longmemeval: dict[str, Any],
    p3_natural_adaptive_quota_mrcr: dict[str, Any],
    p3_natural_adaptive_quota_suite: dict[str, Any],
    p3_natural: dict[str, Any],
    p3_safety: dict[str, Any],
    p3_natural_safety: dict[str, Any],
    p3_ifeval: dict[str, Any],
    p3_longsafety: dict[str, Any],
    p4_500k_context: dict[str, Any],
    p4_reference_systems: dict[str, Any],
    p4_adaptive_systems: dict[str, Any],
    p4_adaptive_production_systems: dict[str, Any],
    p4_production_systems: dict[str, Any],
    inputs: list[dict[str, Any]],
) -> str:
    causal = p2_causal_confirmatory["primary_causal_gate"]
    _causal_passed, causal_pareto_verdict = _validated_confirmatory_causal_pareto(
        p2_causal_confirmatory
    )
    p4_500k = p4_500k_context["audit"]
    p4_500k_correctness = p4_500k_context["correctness"]
    p4_reference = p4_reference_systems["audit"]
    p4_adaptive = p4_adaptive_systems["audit"]
    p4_adaptive_production = p4_adaptive_production_systems["audit"]
    p4_production = p4_production_systems["audit"]
    p3_cross_adaptive = p3_cross_family_adaptive_quota["audit"]
    evidence_lines = "\n".join(
        f"| {row['name']} | {classifications.get(row['name'], 'unverified')} | `{row['sha256']}` |"
        for row in inputs
        if row.get("kind")
        not in {
            "execution-audit",
            "traceability-contract",
            "reproduction-guide",
            "verification-runner",
            "generator",
        }
    )
    execution_lines = "\n".join(
        f"| {row['name']} | verified | `{row['sha256']}` |"
        for row in inputs
        if row.get("kind") == "execution-audit"
    )
    return f"""# Adaptive V4 Memory: paper-grade empirical report

This report is generated only from digest-bound audit artifacts. Conclusion labels are
mechanical and deliberately narrower than the motivating hypothesis.

## Evidence ledger

| Evidence | Classification | SHA-256 |
|---|---|---|
{evidence_lines}

## Execution integrity ledger

| Audit | Status | SHA-256 |
|---|---|---|
{execution_lines}

These audits bind the serial/parallel equivalence probes and their raw child artifacts.
They validate execution semantics and do not receive a scientific conclusion class.
Every P2 statistical summary is also bound to the exact Git-index blobs of the analysis
implementations that produced it; P5 fails closed if those implementations drift.

## Requirement traceability

`table-requirement-traceability.csv` binds all
{len({row["requirement_id"] for row in traceability_rows})} frozen P0-P5 requirements and
completion conditions through {len(traceability_rows)} source links. Trace coverage means that
the relevant evidence, boundary, execution audit, generated output, or final verification
contract is explicit; it does not upgrade any scientific conclusion class. The final local
release gate remains scheduled after package generation. GitHub Actions remains disabled by
user request, is outside the completion gate, and is never reported as passed.

## Experiment volume

- P2 core primary cohort: {p2_core["audit"]["unique_shards"]:,} verified shards and
  5 independently trained seeds. The immutable primary report remains separately auditable.
- P2 core confirmatory cohort: {p2_core_confirmatory["audit"]["unique_shards"]:,}
  verified shards, 9 independently trained seeds, 2 scales, 9 workload families,
  5 contexts, and 1,000 examples per seed-scale-family. Pooling was permitted only after
  identical-contract and disjoint-seed audits passed.
- M5 one-token baseline: {m5_one_token_pilot["audit"]["scales_verified"]} scales,
  {m5_one_token_pilot["audit"]["workloads_per_scale"]} synthetic workloads per scale,
  classified only as a pilot negative result for the tested interface.
- M3 offline learned-risk pilot: {m3_offline_learned_risk_pilot["audit"]["scales_verified"]} scales with
  disjoint train/calibration/test splits and
  {m3_offline_learned_risk_pilot["audit"]["ablation_variants_verified"]} ablations; classified
  as a negative Pareto result. It used final-query probes from a full native pass to build an
  offline replay plan and is explicitly not evidence for deployable online learned lookahead.
- P1 online learned lookahead: {p1_online_learned_lookahead["audit"]["test_shards_verified"]:,}
  held-out shards, {p1_online_learned_lookahead["audit"]["paired_conversations"]:,} paired
  conversations per arm, 5 seeds and 2 scales; the separate exploratory gate passed:
  **{p1_online_learned_lookahead["primary_gate"]["passed"]}**.
- P2 causal primary cohort: {p2_causal["audit"]["unique_shards"]:,} verified factorial
  shards over 5 independently trained seeds; its report remains separately auditable.
- P2 causal confirmatory cohort: {p2_causal_confirmatory["audit"]["unique_shards"]:,}
  verified factorial shards over 9 independently trained seeds;
  {p2_causal_confirmatory["audit"]["quality_execution_counts"]["executed"]:,}
  quality forwards were executed and
  {p2_causal_confirmatory["audit"]["quality_execution_counts"]["reused_exact_config"]:,}
  arm-batches reused an exact byte-identical config; the calibrated+pins versus
  fixed+pins gate passed: **{causal["passed"]}**. The explicit quality-memory
  Pareto verdict is: **{causal_pareto_verdict}**.
- P2 core confirmatory inference: exact enumeration covers
  {p2_core_confirmatory["confirmatory_inference"]["exact_sign_assignments"]} sign
  assignments across 9 independent seeds, giving a minimum attainable two-sided
  seed-level p-value of
  {p2_core_confirmatory["confirmatory_inference"]["minimum_attainable_two_sided_seed_p"]:.6f}.
  The original five-seed cohort and four-seed extension are also reported separately.
- P2 causal confirmatory inference uses nine independent seeds with
  {1 << p2_causal_confirmatory["audit"]["independent_seed_clusters_per_cell"]} exact sign assignments
  and a minimum attainable two-sided p-value of
  {p2_causal_confirmatory["audit"]["minimum_attainable_two_sided_seed_p"]:.6f}. Seed-level p-values
  accompany effect sizes and intervals; they are never the sole success criterion.
- P2 between-seed variation is retained explicitly for every comparison or contrast,
  scale, and memory budget. The seed-variance tables report all independent seed effects,
  their mean, sample standard deviation, and range alongside the separate worst-slice tables.
- P2 supplemental baselines: fixed top-p 0.5/0.8 are evaluated on the complete
  factorial, and the target-aware registered-arm oracle is reported only as a
  non-causal upper bound over {len(p2_causal_confirmatory["offline_oracle_upper_bound"]["registered_arms"])} arms.
- P3 RULER: {p3_ruler["audit"]["completed_cells"]} cells and
  {p3_ruler["audit"]["total_predictions"]:,} predictions on one pinned compatible model.
- P3 cross-family transfer: {p3_cross_family["audit"]["total_predictions"]:,} paired
  Phi-4-mini RULER predictions across 3 lengths and 13 tasks. The Qwen-selected
  50%-KV operating point was transferred without Phi-specific tuning; its frozen
  transfer gate passed: **{p3_cross_family["transfer_gate"]["passed"]}**. This is a
  separately reported model-family transfer cohort, not a second full natural suite.
- P3 Phi adaptive quota: {p3_cross_adaptive["total_predictions"]:,} predictions form
  {p3_cross_adaptive["paired_examples"]:,} fixed+pins/adaptive pairs across the same
  3 lengths and 13 tasks. The Qwen-selected scorer is reused without Phi tuning and
  the result is not pooled with Qwen or interpreted as unchanged synthetic-controller transfer.
- P3 Phi adaptive LongBench-v2 replication:
  {p3_cross_family_adaptive_quota_longbench_v2["audit"]["total_predictions"]:,}
  predictions form
  {p3_cross_family_adaptive_quota_longbench_v2["audit"]["paired_examples"]:,}
  fixed+pins/adaptive pairs over all six frozen reasoning categories. Its confirmation
  gate passed:
  **{p3_cross_family_adaptive_quota_longbench_v2["confirmation_gate"]["passed"]}**.
  This held-out Phi checkpoint reuses the Qwen-selected scorer without Phi tuning,
  is reported separately rather than pooled with Qwen or Phi RULER, and supports no
  claim beyond 128K or continuous adaptive reallocation.
- P3 real-model adaptive quota: {p3_natural_adaptive_quota["audit"]["total_predictions"]:,}
  Qwen3-4B RULER predictions pair fixed+pins with a causal adaptive layer-quota arm at
  exactly the same global KV-token budget; confirmation gate passed:
  **{p3_natural_adaptive_quota["analysis"]["confirmation_gate"]["passed"]}**. This is an
  architecture-compatibility result, not an unchanged synthetic-controller transfer.
- P3 adaptive SCBench replication:
  {p3_natural_adaptive_quota_scbench["audit"]["total_predictions"]:,} Qwen3-4B turn
  predictions cover both shared-context modes and all twelve frozen tasks. The initial
  prefill uses equal global KV tokens and its confirmation gate passed:
  **{p3_natural_adaptive_quota_scbench["confirmation_gate"]["passed"]}**. This does not
  claim adaptive reallocation after prefill or continuous-refresh behavior.
- P3 adaptive LongBench-v2 replication:
  {p3_natural_adaptive_quota_longbench_v2["audit"]["total_predictions"]:,} paired
  Qwen3-4B predictions cover all six frozen reasoning categories with identical token-id
  inputs and equal initial global KV tokens. Its confirmation gate passed:
  **{p3_natural_adaptive_quota_longbench_v2["confirmation_gate"]["passed"]}**. Secondary
  sub-domain, difficulty, and length slices are descriptive only, and this result does
  not claim continuous adaptive reallocation.
- P3 adaptive LongMemEval generation cohort:
  {p3_natural_adaptive_quota_longmemeval["audit"]["total_predictions"]:,} paired
  Qwen3-4B response attempts cover all 500 frozen questions with exact token-id pairing
  and equal initial global KV tokens. Successful raw generations and physical audits are
  retained, but the official GPT-4o judge remains blocked. Consequently its quality
  classification and confirmation gate remain **unverified**; no proxy score or zero-score
  substitution is used, and it is excluded from the scored adaptive natural-suite gate.
- P3 adaptive MRCR replication:
  {p3_natural_adaptive_quota_mrcr["audit"]["total_predictions"]:,} Qwen3-4B
  predictions cover 2/4/8 needles and all five frozen token bins through 128K.
  Exact token-id inputs and initial global KV tokens are paired; its confirmation
  gate passed: **{p3_natural_adaptive_quota_mrcr["confirmation_gate"]["passed"]}**.
  Every needle/bin cell remains visible, and this result does not claim continuous
  adaptive reallocation or contexts beyond 128K.
- P3 adaptive natural suite: the four digest-bound component audits account for
  {p3_natural_adaptive_quota_suite["audit"]["total_predictions"]:,} predictions without
  pooling heterogeneous scores or p-values. Its preregistered broad-transfer gate passed:
  **{p3_natural_adaptive_quota_suite["suite_confirmation_gate"]["passed"]}** with
  {p3_natural_adaptive_quota_suite["suite_confirmation_gate"]["component_gates_passed"]}/4
  component gates and
  {p3_natural_adaptive_quota_suite["suite_confirmation_gate"]["nonnegative_overall_effects"]}/4
  nonnegative overall effects. LongMemEval remains excluded because its official judge is
  unavailable; no auxiliary metric is substituted.
- P3 natural suite: {p3_natural["audit"]["benchmarks_terminal"]} terminal benchmarks and
  at least {p3_natural["audit"]["minimum_protocol_examples_accounted_per_arm"]:,}
  examples accounted per required arm. The fixed arm was selected before any Qwen3-4B
  outcome as the best of four frozen Qwen3-1.7B RULER candidates at 50% KV over
  8K/16K/32K, then transferred unchanged. "Strongest" is restricted to that selection
  grid and is not a claim of global dominance on Qwen3-4B or every natural benchmark.
- External DSA comparators: FlashMemory-DeepSeek-V4 and IndexCache are not Qwen3-compatible
  baselines and remain unverified rather than projected. IndexCache is pinned at
  `08d22d69b1aa2aa0a3de23df6d6b88dbd5b5d044` and may be compared only on its supported
  DeepSeek Sparse Attention runtime; the official FlashMemory contract remains separately blocked.
- P3 safety stress: {p3_safety["audit"]["examples_accounted_per_arm"]:,} examples per arm,
  {p3_safety["audit"]["families_terminal"]} families, and
  {p3_safety["audit"]["contexts_terminal"]} context lengths with paired inputs.
- P3 natural safety suite: {p3_natural_safety["audit"]["required_arms"]} paired arms;
  LongSafety generation and IFEval official scoring are terminal, while the paid
  LongSafety judge remains **{p3_natural_safety["audit"]["longsafety_official_judge_status"]}**.
- P3 IFEval control: {p3_ifeval["audit"]["expected_prompts_per_arm"]:,} officially scored
  prompts per required arm.
- P3 LongSafety: {p3_longsafety["audit"]["expected_generations_total"]:,} digest-bound
  generations; official paid judge status is **{p3_longsafety["audit"]["official_judge_status"]}**,
  so no comparative LongSafety safety score is claimed.
- P4 500K feasibility: {p4_500k["terminal_policy_attempts"]} terminal scale-policy
  attempts, {p4_500k["successful_policy_attempts"]} successful and
  {p4_500k["failed_policy_attempts"]} failed. This single-attempt preflight carries
  no performance claim. Recorded POSIX timer deadlines ranged from
  {p4_500k["minimum_cell_timeout_seconds"]:,.0f} to
  {p4_500k["maximum_cell_timeout_seconds"]:,.0f} seconds and do not establish native
  CUDA-call preemption; {p4_500k_correctness["scales_with_both_policies_successful"]}
  paired-success scales had prediction equality
  **{p4_500k_correctness["all_successful_pair_predictions_identical"]}**.
- P4 reference systems: {p4_reference["terminal_cells"]} terminal serial-interleaved cells,
  {p4_reference["complete_cells"]} complete, {p4_reference["partial_cells"]} partial, and
  {p4_reference["failed_cells"]} failed. Digest-bound per-cell POSIX timer deadlines ranged
  from {p4_reference["minimum_cell_timeout_seconds"]:,.0f} to
  {p4_reference["maximum_cell_timeout_seconds"]:,.0f} seconds; this does not claim that a
  Python signal preempts an uninterruptible native CUDA call.
- P4 adaptive systems: {p4_adaptive["terminal_cells"]} terminal serial-interleaved
  `fixed+pins` versus `calibrated+pins` cells across both 2x/4x budgets;
  {p4_adaptive["complete_cells"]} complete, {p4_adaptive["partial_cells"]} partial,
  and {p4_adaptive["failed_cells"]} failed. The P2 gate outcome was recorded but did not
  select cells. Controller time, physical hot budgets, raw latency, HBM, and transfer
  metrics are independently audited; this table is not an actual-concurrency claim.
- P4 adaptive production systems: {p4_adaptive_production["terminal_cells"]} terminal
  simultaneous static-batch `fixed+pins` versus `calibrated+pins` cells;
  {p4_adaptive_production["complete_cells"]} complete,
  {p4_adaptive_production["partial_cells"]} partial, and
  {p4_adaptive_production["failed_cells"]} failed. The audit verifies paired controller
  schedules, physical hot-memory equality, raw request/decode timestamps, 10,000 paired
  bootstrap resamples, and Holm correction. It does not claim dynamic arrivals,
  continuous admission, fused kernels, kernel-aware residency layout, position-aware
  recomputation costs, or an external production runtime.
- P4 production systems: {p4_production["terminal_cells"]} terminal actual-concurrency cells,
  {p4_production["complete_cells"]} complete, {p4_production["partial_cells"]} partial, and
  {p4_production["failed_cells"]} failed. Each adapter cell runs in a subprocess under a
  digest-bound hard deadline ranging from
  {p4_production["minimum_cell_timeout_seconds"]:,.0f} to
  {p4_production["maximum_cell_timeout_seconds"]:,.0f} seconds.
  Warmup accounting availability was recorded for every terminal cell; accounting was
  available for every adapter-executed cell: **{p4_production["warmup_accounting_available_all_adapter_cells"]}**.
  {p4_production["warmup_accounting_unavailable_cells"]} orchestrator-failure cells retain
  explicit unavailable/null warmup fields rather than invented zero counts.
  Allocator/device HBM remains mandatory for successful runs; process-total HBM was available
  for {p4_production["process_total_hbm_measured_runs"]:,} measured policy runs and explicitly
  unavailable for {p4_production["process_total_hbm_unavailable_runs"]:,}, with no zero or proxy
  imputation.
  The long-form P4 metric tables retain run-level distributions (mean, standard deviation,
  p50/p95/p99, minimum, and maximum) plus paired bootstrap effects for every registered
  latency, throughput, HBM, fragmentation, cache, transfer, miss, and controller metric.
  These measurements do not verify non-contiguous fused-attention layout costs or
  position-aware miss recomputation; those remain explicit external-runtime blockers.

## Digest-bound figures

![Causal effect with corrected intervals](figure-p2-causal-effect.svg)

The causal figure reports the four preregistered scale-budget cells without pooling them
into a single favorable average. Its interval and point data are embedded in the SVG
metadata and bound to the audited causal matrix. The accompanying P2 tables expose both
core comparisons and preregistered causal contrasts at pooled, seed, and scale-family levels,
including Holm-corrected family inference, worst slices, measured physical-memory matching,
and the target-aware oracle separately.

![Natural benchmark paired quality](figure-p3-natural-quality.svg)

The natural-language figure reports every benchmark separately, scores all operational
failures as zero, and uses paired bootstrap intervals over the frozen example set. The
adjacent CSV tables retain arm-level failure, latency, peak-HBM, and hot-memory summaries.

![Production latency, throughput, and memory trade-offs](figure-p4-production-tradeoffs.svg)

The production figure reports mean and full observed cell range for each context-metric
pair. Failed cells remain in the terminal counts in the subtitle and are never imputed as
measured ratios.

## Claim boundary

The P2 result is synthetic Tier-S evidence. A failed causal gate bounds only the tested
controller family. P3 quality and synthetic safety-retention results are transfer evidence
for pinned Qwen3 snapshots, not comprehensive safety certification, official DeepSeek-V4
evidence, or model-population inference. P4 reference evidence is single-accelerator and
serial-interleaved. The 500K result is feasibility-only on Tier-S scales; the checked
production adapter is static continuous batching, not an
external dynamic/fused/multi-GPU serving runtime. Official DeepSeek-V4 stays unverified until its
frozen resource contract is satisfied.

## Reproduction

The dependency-ordered commands, resume rules, failure policy, external resource boundary,
and final local release checks are frozen in [the reproduction guide](reproduction-guide.md).
The CSV tables next to this report are generated from the same frozen audits. Their digests,
the input digests, source commit, protocol manifests, and reproduction guide are recorded in
`artifact-index.json`; missing or incomplete evidence causes generation to fail rather than
being imputed. GitHub Actions remains disabled by user request and is not reported as passed;
the digest-bound local release gate is the final source-verification contract.
"""


def build_package(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    _require(
        manifest.get("experiment_id") == "adaptive-v4-memory-p5-paper-package-v1",
        "Wrong P5 package manifest.",
    )
    loaded: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = [_paper_package_generator_input()]
    reproduction_path = Path(manifest.get("reproduction_guide", {}).get("path", ""))
    reproduction_guide = _validate_reproduction_guide(reproduction_path)
    inputs.append(
        {
            "name": "reproduction_guide",
            "kind": "reproduction-guide",
            "path": str(reproduction_path),
            "sha256": sha256(reproduction_path),
        }
    )
    traceability_contract = manifest.get("requirement_traceability", {})
    traceability_path = Path(traceability_contract.get("path", ""))
    traceability = _load(traceability_path)
    _require(
        traceability.get("experiment_id") == traceability_contract.get("experiment_id"),
        "P5 requirement traceability contract drifted.",
    )
    inputs.append(
        {
            "name": "requirement_traceability",
            "kind": "traceability-contract",
            "path": str(traceability_path),
            "sha256": sha256(traceability_path),
        }
    )
    release_runner = Path(
        traceability["verification_contracts"]["final-local-release-gate"]["runner"]
    )
    _require(release_runner.is_file(), f"Missing P5 release-gate runner: {release_runner}")
    inputs.append(
        {
            "name": "local_release_gate_runner",
            "kind": "verification-runner",
            "path": str(release_runner),
            "sha256": sha256(release_runner),
        }
    )
    for name, contract in manifest["evidence"].items():
        path = Path(contract["path"])
        loaded[name] = _validate_evidence(name, path, contract)
        inputs.append({"name": name, "path": str(path), "sha256": sha256(path)})
    for name, contract in manifest["execution_audits"].items():
        path = Path(contract["path"])
        _validate_execution_audit(name, path, contract)
        inputs.append(
            {
                "name": name,
                "kind": "execution-audit",
                "path": str(path),
                "sha256": sha256(path),
            }
        )
    _require(
        set(manifest["boundary_manifests"]) == set(BOUNDARY_EXPERIMENT_IDS),
        "P5 boundary manifest set drifted.",
    )
    for name, raw_path in manifest["boundary_manifests"].items():
        path = Path(raw_path)
        _validate_boundary_manifest(name, path)
        inputs.append({"name": name, "path": str(path), "sha256": sha256(path)})

    classes = classify_evidence(
        loaded["p2_core"],
        loaded["m5_one_token_pilot"],
        loaded["m3_offline_learned_risk_pilot"],
        loaded["p1_online_learned_lookahead"],
        loaded["p2_causal"],
        loaded["p3_ruler"],
        loaded["p3_natural"],
        loaded["p3_safety"],
        loaded["p3_natural_safety"],
        loaded["p3_ifeval"],
        loaded["p3_longsafety"],
        loaded["p4_500k_context"],
        loaded["p4_reference_systems"],
        loaded["p4_production_systems"],
        loaded["p4_adaptive_systems"],
        p4_adaptive_production_systems=loaded["p4_adaptive_production_systems"],
        p3_cross_family=loaded["p3_cross_family"],
        p3_cross_family_adaptive_quota=loaded["p3_cross_family_adaptive_quota"],
        p3_cross_family_adaptive_quota_longbench_v2=loaded[
            "p3_cross_family_adaptive_quota_longbench_v2"
        ],
        p3_natural_adaptive_quota=loaded["p3_natural_adaptive_quota"],
        p3_natural_adaptive_quota_scbench=loaded[
            "p3_natural_adaptive_quota_scbench"
        ],
        p3_natural_adaptive_quota_longbench_v2=loaded[
            "p3_natural_adaptive_quota_longbench_v2"
        ],
        p3_natural_adaptive_quota_longmemeval=loaded[
            "p3_natural_adaptive_quota_longmemeval"
        ],
        p3_natural_adaptive_quota_mrcr=loaded["p3_natural_adaptive_quota_mrcr"],
        p3_natural_adaptive_quota_suite=loaded[
            "p3_natural_adaptive_quota_suite"
        ],
    )
    classes["p2_core_confirmatory"] = _classify_validated_confirmatory_core(
        loaded["p2_core_confirmatory"]
    )
    classes["p2_causal_confirmatory"] = _classify_validated_confirmatory_causal(
        loaded["p2_causal_confirmatory"]
    )
    _validate_required_classifications(classes, manifest.get("required_classifications"))
    traceability_rows = _traceability_rows(traceability, manifest, classes)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reproduction-guide.md").write_text(reproduction_guide)
    evidence_rows = [
        {**row, "classification": classes.get(row["name"], "unverified")}
        for row in inputs
        if row.get("kind")
        not in {
            "execution-audit",
            "traceability-contract",
            "reproduction-guide",
            "verification-runner",
            "generator",
        }
    ]
    _write_csv(
        output_root / "table-evidence.csv",
        evidence_rows,
        ["name", "classification", "path", "sha256"],
    )
    _write_csv(
        output_root / "table-requirement-traceability.csv",
        traceability_rows,
        [
            "requirement_id",
            "phase",
            "requirement",
            "source_kind",
            "source_name",
            "binding_status",
            "scientific_classification",
        ],
    )
    inference_resolution = _p2_inference_resolution_rows(
        loaded["p2_core_confirmatory"], loaded["p2_causal_confirmatory"]
    )
    _write_csv(
        output_root / "table-p2-inference-resolution.csv",
        inference_resolution,
        list(inference_resolution[0]),
    )
    quality = _p2_quality_rows(loaded["p2_core_confirmatory"])
    _write_csv(output_root / "table-p2-quality-gate.csv", quality, list(quality[0]))
    p2_core_tables = {
        "table-p2-core-policy-summary.csv": [
            _flatten_json_row(row) for row in loaded["p2_core_confirmatory"]["policy_summary"]
        ],
        "table-p2-core-effects.csv": _p2_core_effect_rows(loaded["p2_core_confirmatory"]),
        "table-p2-core-family-effects.csv": _p2_core_family_rows(loaded["p2_core_confirmatory"]),
        "table-p2-core-seed-effects.csv": _p2_core_seed_rows(loaded["p2_core_confirmatory"]),
        "table-p2-core-seed-variance.csv": _p2_core_seed_variance_rows(
            loaded["p2_core_confirmatory"]
        ),
        "table-p2-core-worst-slices.csv": _p2_core_worst_slice_rows(loaded["p2_core_confirmatory"]),
    }
    for name, rows in p2_core_tables.items():
        _write_csv(output_root / name, rows, _field_union(rows))
    learned = _learned_lookahead_rows(loaded["p1_online_learned_lookahead"])
    _write_csv(
        output_root / "table-p1-online-learned-lookahead-gate.csv",
        learned,
        list(learned[0]),
    )
    causal = _causal_rows(loaded["p2_causal_confirmatory"])
    _write_csv(output_root / "table-p2-causal-gate.csv", causal, list(causal[0]))
    p2_causal_tables = {
        "table-p2-causal-contrasts.csv": _causal_contrast_rows(loaded["p2_causal_confirmatory"]),
        "table-p2-causal-family-effects.csv": _causal_family_rows(loaded["p2_causal_confirmatory"]),
        "table-p2-causal-seed-effects.csv": _causal_seed_rows(loaded["p2_causal_confirmatory"]),
        "table-p2-causal-seed-variance.csv": _causal_seed_variance_rows(
            loaded["p2_causal_confirmatory"]
        ),
        "table-p2-causal-worst-slices.csv": _causal_worst_slice_rows(
            loaded["p2_causal_confirmatory"]
        ),
        "table-p2-causal-physical-memory.csv": _causal_physical_memory_rows(
            loaded["p2_causal_confirmatory"]
        ),
        "table-p2-causal-offline-oracle.csv": _causal_oracle_rows(loaded["p2_causal_confirmatory"]),
    }
    for name, rows in p2_causal_tables.items():
        _write_csv(output_root / name, rows, _field_union(rows))
    p3 = _p3_rows(loaded["p3_ruler"])
    _write_csv(output_root / "table-p3-ruler-cells.csv", p3, list(p3[0]))
    p3_cross_task_length = _p3_cross_family_task_length_rows(loaded["p3_cross_family"])
    _write_csv(
        output_root / "table-p3-cross-family-task-length.csv",
        p3_cross_task_length,
        _field_union(p3_cross_task_length),
    )
    p3_cross_length = _p3_cross_family_length_rows(loaded["p3_cross_family"])
    _write_csv(
        output_root / "table-p3-cross-family-length-inference.csv",
        p3_cross_length,
        _field_union(p3_cross_length),
    )
    p3_cross_adaptive_task_length = _p3_natural_adaptive_task_length_rows(
        loaded["p3_cross_family_adaptive_quota"]
    )
    _write_csv(
        output_root / "table-p3-cross-family-adaptive-task-length.csv",
        p3_cross_adaptive_task_length,
        _field_union(p3_cross_adaptive_task_length),
    )
    p3_cross_adaptive_length = _p3_natural_adaptive_length_rows(
        loaded["p3_cross_family_adaptive_quota"]
    )
    _write_csv(
        output_root / "table-p3-cross-family-adaptive-length-inference.csv",
        p3_cross_adaptive_length,
        _field_union(p3_cross_adaptive_length),
    )
    p3_cross_adaptive_layers = _p3_natural_adaptive_layer_rows(
        loaded["p3_cross_family_adaptive_quota"]
    )
    _write_csv(
        output_root / "table-p3-cross-family-adaptive-layer-distributions.csv",
        p3_cross_adaptive_layers,
        _field_union(p3_cross_adaptive_layers),
    )
    p3_cross_adaptive_longbench_summary = _p3_adaptive_longbench_summary_rows(
        loaded["p3_cross_family_adaptive_quota_longbench_v2"]
    )
    _write_csv(
        output_root / "table-p3-cross-family-adaptive-longbench-summary.csv",
        p3_cross_adaptive_longbench_summary,
        _field_union(p3_cross_adaptive_longbench_summary),
    )
    p3_cross_adaptive_longbench_categories = _p3_adaptive_longbench_category_rows(
        loaded["p3_cross_family_adaptive_quota_longbench_v2"]
    )
    _write_csv(
        output_root / "table-p3-cross-family-adaptive-longbench-categories.csv",
        p3_cross_adaptive_longbench_categories,
        _field_union(p3_cross_adaptive_longbench_categories),
    )
    p3_cross_adaptive_longbench_slices = _p3_adaptive_longbench_slice_rows(
        loaded["p3_cross_family_adaptive_quota_longbench_v2"]
    )
    _write_csv(
        output_root / "table-p3-cross-family-adaptive-longbench-slices.csv",
        p3_cross_adaptive_longbench_slices,
        _field_union(p3_cross_adaptive_longbench_slices),
    )
    p3_adaptive_task_length = _p3_natural_adaptive_task_length_rows(
        loaded["p3_natural_adaptive_quota"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-task-length.csv",
        p3_adaptive_task_length,
        _field_union(p3_adaptive_task_length),
    )
    p3_adaptive_length = _p3_natural_adaptive_length_rows(loaded["p3_natural_adaptive_quota"])
    _write_csv(
        output_root / "table-p3-natural-adaptive-length-inference.csv",
        p3_adaptive_length,
        _field_union(p3_adaptive_length),
    )
    p3_adaptive_layers = _p3_natural_adaptive_layer_rows(loaded["p3_natural_adaptive_quota"])
    _write_csv(
        output_root / "table-p3-natural-adaptive-layer-distributions.csv",
        p3_adaptive_layers,
        _field_union(p3_adaptive_layers),
    )
    p3_adaptive_scbench_summary = _p3_adaptive_scbench_summary_rows(
        loaded["p3_natural_adaptive_quota_scbench"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-scbench-summary.csv",
        p3_adaptive_scbench_summary,
        _field_union(p3_adaptive_scbench_summary),
    )
    p3_adaptive_scbench_mode_task = _p3_adaptive_scbench_mode_task_rows(
        loaded["p3_natural_adaptive_quota_scbench"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-scbench-mode-task.csv",
        p3_adaptive_scbench_mode_task,
        _field_union(p3_adaptive_scbench_mode_task),
    )
    p3_adaptive_longbench_summary = _p3_adaptive_longbench_summary_rows(
        loaded["p3_natural_adaptive_quota_longbench_v2"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-longbench-summary.csv",
        p3_adaptive_longbench_summary,
        _field_union(p3_adaptive_longbench_summary),
    )
    p3_adaptive_longbench_categories = _p3_adaptive_longbench_category_rows(
        loaded["p3_natural_adaptive_quota_longbench_v2"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-longbench-categories.csv",
        p3_adaptive_longbench_categories,
        _field_union(p3_adaptive_longbench_categories),
    )
    p3_adaptive_longbench_slices = _p3_adaptive_longbench_slice_rows(
        loaded["p3_natural_adaptive_quota_longbench_v2"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-longbench-slices.csv",
        p3_adaptive_longbench_slices,
        _field_union(p3_adaptive_longbench_slices),
    )
    p3_adaptive_longmemeval_summary = _p3_adaptive_longmemeval_summary_rows(
        loaded["p3_natural_adaptive_quota_longmemeval"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-longmemeval-summary.csv",
        p3_adaptive_longmemeval_summary,
        _field_union(p3_adaptive_longmemeval_summary),
    )
    p3_adaptive_longmemeval_question_types = (
        _p3_adaptive_longmemeval_question_type_rows(
            loaded["p3_natural_adaptive_quota_longmemeval"]
        )
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-longmemeval-question-types.csv",
        p3_adaptive_longmemeval_question_types,
        _field_union(p3_adaptive_longmemeval_question_types),
    )
    p3_adaptive_mrcr_summary = _p3_adaptive_mrcr_summary_rows(
        loaded["p3_natural_adaptive_quota_mrcr"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-mrcr-summary.csv",
        p3_adaptive_mrcr_summary,
        _field_union(p3_adaptive_mrcr_summary),
    )
    p3_adaptive_mrcr_cells = _p3_adaptive_mrcr_cell_rows(
        loaded["p3_natural_adaptive_quota_mrcr"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-mrcr-cells.csv",
        p3_adaptive_mrcr_cells,
        _field_union(p3_adaptive_mrcr_cells),
    )
    p3_adaptive_suite = _p3_adaptive_suite_rows(
        loaded["p3_natural_adaptive_quota_suite"]
    )
    _write_csv(
        output_root / "table-p3-natural-adaptive-suite.csv",
        p3_adaptive_suite,
        _field_union(p3_adaptive_suite),
    )
    p3_safety = _p3_safety_rows(loaded["p3_safety"])
    _write_csv(
        output_root / "table-p3-safety-slices.csv",
        p3_safety,
        list(p3_safety[0]),
    )
    p3_natural_arms = _p3_natural_arm_rows(loaded["p3_natural"])
    _write_csv(
        output_root / "table-p3-natural-benchmark-arms.csv",
        p3_natural_arms,
        list(p3_natural_arms[0]),
    )
    p3_natural_contrasts = _p3_natural_contrast_rows(loaded["p3_natural"])
    _write_csv(
        output_root / "table-p3-natural-paired-contrasts.csv",
        p3_natural_contrasts,
        list(p3_natural_contrasts[0]),
    )
    p4_500k = _p4_500k_rows(loaded["p4_500k_context"])
    _write_csv(
        output_root / "table-p4-500k-context.csv",
        p4_500k,
        list(p4_500k[0]),
    )
    p4_reference = _p4_rows(loaded["p4_reference_systems"])
    p4_adaptive = _p4_adaptive_rows(loaded["p4_adaptive_systems"])
    p4_adaptive_production = _p4_adaptive_production_rows(
        loaded["p4_adaptive_production_systems"]
    )
    p4_production = _p4_rows(loaded["p4_production_systems"])
    p4_fields = [
        "scale",
        "context",
        "generation",
        "profile",
        "batch",
        "active_requests",
        "concurrency",
        "status",
        "paired_repetitions",
        "cell_timeout_seconds",
        "warmup_accounting_available",
        "warmup_repetitions_attempted",
        "warmup_paired_repetitions_completed",
        "warmup_failures",
        "resident_ttft_p95_ms_mean",
        "tiered_ttft_p95_ms_mean",
        "resident_throughput_mean",
        "tiered_throughput_mean",
        "resident_peak_hbm_mean",
        "tiered_peak_hbm_mean",
        "failure",
    ]
    _write_csv(
        output_root / "table-p4-reference-system-cells.csv",
        p4_reference,
        p4_fields,
    )
    _write_csv(
        output_root / "table-p4-production-system-cells.csv",
        p4_production,
        p4_fields,
    )
    _write_csv(
        output_root / "table-p4-adaptive-system-cells.csv",
        p4_adaptive,
        [
            "scale",
            "budget",
            "context",
            "generation",
            "profile",
            "batch",
            "active_requests",
            "status",
            "paired_repetitions",
            "cell_timeout_seconds",
            "warmup_repetitions_attempted",
            "warmup_paired_repetitions_completed",
            "warmup_failures",
            "fixed_ttft_p95_ms_mean",
            "calibrated_ttft_p95_ms_mean",
            "fixed_throughput_mean",
            "calibrated_throughput_mean",
            "fixed_peak_hbm_mean",
            "calibrated_peak_hbm_mean",
            "fixed_controller_time_ns_mean",
            "calibrated_controller_time_ns_mean",
            "prediction_digest_mismatches",
            "failure",
        ],
    )
    _write_csv(
        output_root / "table-p4-adaptive-production-system-cells.csv",
        p4_adaptive_production,
        _field_union(p4_adaptive_production),
    )
    p4_metric_fields = [
        "scale",
        "context",
        "generation",
        "profile",
        "batch",
        "active_requests",
        "concurrency",
        "status",
        "metric",
        "policy",
        "observations",
        "mean",
        "sample_standard_deviation",
        "p50",
        "p95",
        "p99",
        "minimum",
        "maximum",
        "paired_observations",
        "mean_ratio_tiered_over_resident",
        "paired_tiered_minus_resident",
    ]
    _write_csv(
        output_root / "table-p4-reference-system-metrics.csv",
        _p4_metric_rows(loaded["p4_reference_systems"]),
        p4_metric_fields,
    )
    _write_csv(
        output_root / "table-p4-production-system-metrics.csv",
        _p4_metric_rows(loaded["p4_production_systems"]),
        p4_metric_fields,
    )
    _write_csv(
        output_root / "table-p4-adaptive-system-metrics.csv",
        _p4_adaptive_metric_rows(loaded["p4_adaptive_systems"]),
        [
            "scale",
            "budget",
            "context",
            "generation",
            "profile",
            "batch",
            "active_requests",
            "status",
            "metric",
            "policy",
            "observations",
            "mean",
            "sample_standard_deviation",
            "p50",
            "p95",
            "p99",
            "minimum",
            "maximum",
            "paired_observations",
            "mean_ratio_calibrated_over_fixed",
            "paired_calibrated_minus_fixed",
        ],
    )
    p4_adaptive_production_metrics = _p4_adaptive_production_metric_rows(
        loaded["p4_adaptive_production_systems"]
    )
    _write_csv(
        output_root / "table-p4-adaptive-production-system-metrics.csv",
        p4_adaptive_production_metrics,
        [
            "scale",
            "budget",
            "context",
            "generation",
            "profile",
            "batch",
            "concurrency",
            "status",
            "metric",
            "policy",
            "observations",
            "mean",
            "sample_standard_deviation",
            "p50",
            "p95",
            "p99",
            "minimum",
            "maximum",
            "paired_repetitions",
            "paired_calibrated_minus_fixed",
        ],
    )
    _write_p2_causal_figure(
        output_root / "figure-p2-causal-effect.svg",
        loaded["p2_causal_confirmatory"],
    )
    _write_p3_natural_figure(output_root / "figure-p3-natural-quality.svg", loaded["p3_natural"])
    _write_p4_tradeoff_figure(
        output_root / "figure-p4-production-tradeoffs.svg",
        loaded["p4_production_systems"],
    )
    report = _report(
        classifications=classes,
        traceability_rows=traceability_rows,
        p2_core=loaded["p2_core"],
        p2_core_confirmatory=loaded["p2_core_confirmatory"],
        m5_one_token_pilot=loaded["m5_one_token_pilot"],
        m3_offline_learned_risk_pilot=loaded["m3_offline_learned_risk_pilot"],
        p1_online_learned_lookahead=loaded["p1_online_learned_lookahead"],
        p2_causal=loaded["p2_causal"],
        p2_causal_confirmatory=loaded["p2_causal_confirmatory"],
        p3_ruler=loaded["p3_ruler"],
        p3_cross_family=loaded["p3_cross_family"],
        p3_cross_family_adaptive_quota=loaded["p3_cross_family_adaptive_quota"],
        p3_cross_family_adaptive_quota_longbench_v2=loaded[
            "p3_cross_family_adaptive_quota_longbench_v2"
        ],
        p3_natural_adaptive_quota=loaded["p3_natural_adaptive_quota"],
        p3_natural_adaptive_quota_scbench=loaded[
            "p3_natural_adaptive_quota_scbench"
        ],
        p3_natural_adaptive_quota_longbench_v2=loaded[
            "p3_natural_adaptive_quota_longbench_v2"
        ],
        p3_natural_adaptive_quota_longmemeval=loaded[
            "p3_natural_adaptive_quota_longmemeval"
        ],
        p3_natural_adaptive_quota_mrcr=loaded["p3_natural_adaptive_quota_mrcr"],
        p3_natural_adaptive_quota_suite=loaded[
            "p3_natural_adaptive_quota_suite"
        ],
        p3_natural=loaded["p3_natural"],
        p3_safety=loaded["p3_safety"],
        p3_natural_safety=loaded["p3_natural_safety"],
        p3_ifeval=loaded["p3_ifeval"],
        p3_longsafety=loaded["p3_longsafety"],
        p4_500k_context=loaded["p4_500k_context"],
        p4_reference_systems=loaded["p4_reference_systems"],
        p4_adaptive_systems=loaded["p4_adaptive_systems"],
        p4_adaptive_production_systems=loaded["p4_adaptive_production_systems"],
        p4_production_systems=loaded["p4_production_systems"],
        inputs=inputs,
    )
    (output_root / "paper-report.md").write_text(report)
    generated = [
        output_root / name for name in manifest["generated_files"] if name != "artifact-index.json"
    ]
    commit, dirty = _clean_source()
    _require(not dirty, "P5 package generation requires a clean source tree.")
    _require(
        inputs[0] == _paper_package_generator_input(),
        "P5 generator changed while outputs were being generated.",
    )
    index = {
        "schema_version": 1,
        "experiment_id": "adaptive-v4-memory-p5-artifact-index-v1",
        "source": {"commit": commit, "dirty": False},
        "package_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "inputs": inputs,
        "classifications": classes,
        "generated": [{"path": str(path), "sha256": sha256(path)} for path in generated],
    }
    target = output_root / "artifact-index.json"
    target.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict P5 paper package.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p5-paper-package-v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p5"),
    )
    args = parser.parse_args()
    index = build_package(args.manifest, args.output_root)
    print(json.dumps(index["classifications"], sort_keys=True))


if __name__ == "__main__":
    main()
