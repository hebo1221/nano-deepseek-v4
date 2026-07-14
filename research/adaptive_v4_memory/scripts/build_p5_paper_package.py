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
    "verification-contract",
}
FINAL_RELEASE_COMMANDS = [
    ".venv/bin/ruff check nano_deepseek_v4 research/adaptive_v4_memory/scripts tests",
    ".venv/bin/mypy nano_deepseek_v4 research/adaptive_v4_memory/scripts",
    ".venv/bin/pytest -q",
    ".venv/bin/python -m build",
    ".venv/bin/twine check dist/*",
]
REPRODUCTION_REQUIRED_MARKERS = [
    "run_p2_core_parallel.py --scale s55 --workers 3",
    "run_p2_core_parallel.py --scale s151 --workers 3",
    "summarize_p2_core_matrix.py",
    "run_p2_causal_prerequisites.py",
    "run_p2_causal_parallel.py --workers 3",
    "summarize_p2_causal_factorial.py",
    "run_p3_natural_ruler.py",
    "run_p3_scbench.py",
    "run_p3_longbench_v2.py",
    "run_p3_longmemeval.py",
    "run_p3_mrcr.py",
    "run_p3_safety_stress.py",
    "run_p4_500k_context_preflight.py",
    "run_p4_systems_matrix.py",
    "run_p4_production_systems_matrix.py",
    "run_p1_online_lookahead_parallel.py --workers 3",
    "build_p5_paper_package.py",
    "run_p5_release_gate.py",
    *FINAL_RELEASE_COMMANDS,
    "A successful final GitHub Actions CI run remains mandatory before goal completion",
    "Official DeepSeek-V4 boundary",
]
BOUNDARY_EXPERIMENT_IDS = {
    "paper_grade_study": "adaptive-v4-memory-paper-grade-v1",
    "experiment_scale_audit": "adaptive-v4-memory-experiment-scale-audit-v1",
    "p2_causal_factorial": "p2-causal-factorial-v1",
    "online_learned_lookahead": "p1-online-learned-lookahead-v1",
    "p3_ruler": "p3-ruler-qwen3-1.7b-v1",
    "natural_suite": "p3-natural-language-suite-v1",
    "safety_stress": "p3-qwen3-4b-safety-stress-v1",
    "natural_safety": "p3-qwen3-4b-natural-safety-v1",
    "p4_500k_context": "p4-500k-context-preflight-v1",
    "p4_reference_systems": "p4-reference-systems-matrix-v1",
    "p4_production_systems": "p4-production-systems-matrix-v1",
    "official_deepseek_v4": "p3-official-flashmemory-deepseek-v4-v1",
    "production_runtime_blocker": "p4-production-resource-blocker-v1",
}

SCALE_AUDIT_SOURCE_MANIFESTS = {
    "study": Path("research/adaptive_v4_memory/manifests/paper-grade-study-v1.json"),
    "causal": Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json"),
    "online": Path("research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json"),
    "ruler": Path("research/adaptive_v4_memory/manifests/p3-ruler-qwen3-1.7b-v1.json"),
    "natural": Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    "safety": Path("research/adaptive_v4_memory/manifests/p3-safety-stress-v1.json"),
    "natural_safety": Path("research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"),
    "p4_reference": Path(
        "research/adaptive_v4_memory/manifests/p4-reference-systems-matrix-v1.json"
    ),
    "p4_preflight": Path("research/adaptive_v4_memory/manifests/p4-500k-context-preflight-v1.json"),
    "p4_production": Path(
        "research/adaptive_v4_memory/manifests/p4-production-systems-matrix-v1.json"
    ),
}
SCALE_AUDIT_CORE_DESIGN = Path(
    "research/adaptive_v4_memory/scripts/evaluate_p2_core_shard.py"
)


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
                        isinstance(argument, ast.Constant)
                        and isinstance(argument.value, int)
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
    reference_batches = sorted(
        {row.get("batch") for row in reference.get("load_profiles", [])}
    )
    reference_loads = sorted(
        {row.get("active_requests") for row in reference.get("load_profiles", [])}
    )
    production_batches = sorted(
        {row.get("batch") for row in production.get("load_profiles", [])}
    )
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
        "seed_cluster_bootstrap_resamples": study.get("statistics", {}).get(
            "bootstrap_resamples"
        ),
        "example_level_role": "paired descriptive precision within a training seed; examples do not increase the number of independent trained-model clusters",
        "p_value_used_as_success_gate": False,
        "interpretation": "The study has high within-seed sample density but only five independent training seeds per scale. Exact and multiplicity-adjusted seed-level p-values are reported, while causal success requires effect direction, corrected seed-cluster intervals, memory matching, and five-of-five seed consistency rather than an unattainable p<0.05 threshold.",
    }
    _require(
        payload.get("inference_resolution") == expected_resolution,
        "Experiment-scale independent-unit resolution drifted.",
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
    if name == "p3_ruler":
        amendments = payload.get("amendments", [])
        observed = payload.get("sequence_gate", {}).get("observed_before_gate", {})
        _require(
            payload.get("status") == "amended_and_frozen_before_execution"
            and isinstance(amendments, list)
            and len(amendments) == 2,
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
            modes.get("mode_a_score_masking", {}).get("full_kv_remains_on_gpu") is True,
            "Official DeepSeek-V4 Mode A memory boundary drifted.",
        )
        _require(
            modes.get("mode_b_pd_disaggregated", {}).get("total_accelerator_slots") == 16,
            "Official DeepSeek-V4 Mode B topology boundary drifted.",
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
            and resources["available_memory_plus_swap_bytes"]
            < base_model["safetensors_bytes"]
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
            len(systems.get("required_metrics", [])) >= 10
            and len(protocol.get("paired_invariants", [])) >= 5
            and len(protocol.get("artifact_contract", [])) >= 4,
            "Official DeepSeek-V4 execution evidence contract is incomplete.",
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
    missing = [
        marker
        for marker in REPRODUCTION_REQUIRED_MARKERS
        if marker not in normalized_guide
    ]
    _require(not missing, f"Reproduction guide is incomplete: {missing}")
    _require(
        "resume-safe" in normalized_guide
        and "Never delete a terminal failure artifact" in normalized_guide
        and "do not report CI as passed" in normalized_guide
        and "does not waive it" in normalized_guide,
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
        and release.get("github_actions") == "required_before_goal_completion"
        and release.get("github_actions_current_status") == "disabled_manually"
        and release.get("github_actions_passed") is False
        and release.get("runner")
        == "research/adaptive_v4_memory/scripts/run_p5_release_gate.py"
        and release.get("output")
        == "artifacts/adaptive_v4_memory/paper_grade/p5/local-release-gate.summary.json"
        and release.get("timing") == "after-final-paper-package-generation",
        "Final local release-gate contract drifted.",
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
) -> dict[str, str]:
    core_passed = any(
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
        and natural_safety_audit.get("ifeval_official_terminal") is True
        and natural_safety_audit.get("ifeval_input_pairing_verified") is True
        and natural_safety_audit.get("ifeval_expected_prompts_per_arm") == 541
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
        and ifeval_audit.get("expected_prompts_per_arm") == 541
    )
    longsafety_audit = p3_longsafety["audit"]
    longsafety_judged = (
        longsafety_audit.get("generation_arms_terminal") is True
        and longsafety_audit.get("input_pairing_verified") is True
        and longsafety_audit.get("generation_failure_accounting_complete") is True
        and longsafety_audit.get("source_implementations_verified") is True
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
        and all(type(value) is int and value >= 0 for value in reference_counts)
        and sum(reference_counts) == P4_EXPECTED_CELLS
    )
    reference_measured = reference_accounted and sum(reference_counts[:2]) > 0
    production_audit = p4_production_systems["audit"]
    production_counts = tuple(
        production_audit.get(field) for field in ("complete_cells", "partial_cells", "failed_cells")
    )
    production_accounted = (
        production_audit.get("terminal_cells") == P4_EXPECTED_CELLS
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
    production_class = (
        "success"
        if production_full and production_external_runtime_verified
        else "bounded-result"
        if production_accounted and production_audit.get("complete_cells", 0) > 0
        else "unverified"
    )
    result = {
        "p2_core": "success" if core_passed else "negative-result",
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
        "p4_production_systems": production_class,
        "production_runtime_blocker": "unverified",
        "official_deepseek_v4": "unverified",
    }
    _require(set(result.values()).issubset(ALLOWED_CLASSES), "Unknown conclusion class.")
    return result


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
                "minimum_attainable_two_sided_seed_p": audit[
                    "minimum_attainable_two_sided_seed_p"
                ],
                "exact_seed_randomization_verified": audit[
                    "exact_seed_randomization_verified"
                ],
                "p_value_used_as_success_gate": audit[
                    "seed_p_values_used_as_success_gate"
                ],
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


def _causal_worst_slice_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contrast, statistics in payload["paired_statistics"].items():
        identity = {
            "contrast": contrast,
            "candidate": statistics["candidate"],
            "comparator": statistics["comparator"],
        }
        rows.append(
            _flatten_json_row(
                {**identity, "scope": "global", **statistics["worst_slice"]}
            )
        )
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
                "warmup_repetitions_attempted": cell[
                    "warmup_repetitions_attempted"
                ],
                "warmup_paired_repetitions_completed": cell[
                    "warmup_paired_repetitions_completed"
                ],
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
                    else json.dumps(
                        cell["policy_status"], sort_keys=True, separators=(",", ":")
                    )
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
                "warmup_accounting_available": failure[
                    "warmup_accounting_available"
                ],
                "warmup_repetitions_attempted": failure[
                    "warmup_repetitions_attempted"
                ],
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
    inputs: list[dict[str, Any]],
) -> str:
    causal = p2_causal["primary_causal_gate"]
    p4_500k = p4_500k_context["audit"]
    p4_500k_correctness = p4_500k_context["correctness"]
    p4_reference = p4_reference_systems["audit"]
    p4_production = p4_production_systems["audit"]
    evidence_lines = "\n".join(
        f"| {row['name']} | {classifications.get(row['name'], 'unverified')} | `{row['sha256']}` |"
        for row in inputs
        if row.get("kind")
        not in {
            "execution-audit",
            "traceability-contract",
            "reproduction-guide",
            "verification-runner",
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

## Requirement traceability

`table-requirement-traceability.csv` binds all
{len({row["requirement_id"] for row in traceability_rows})} frozen P0-P5 requirements and
completion conditions through {len(traceability_rows)} source links. Trace coverage means that
the relevant evidence, boundary, execution audit, generated output, or final verification
contract is explicit; it does not upgrade any scientific conclusion class. The final local
release gate remains scheduled after package generation. GitHub Actions is currently disabled
manually, remains mandatory before goal completion, and is never reported as passed or waived.

## Experiment volume

- P2 core: {p2_core["audit"]["unique_shards"]:,} verified shards, 5 training seeds,
  2 scales, 9 workload families, 5 contexts, and 1,000 examples per
  seed-scale-family.
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
- P2 causal: {p2_causal["audit"]["unique_shards"]:,} verified factorial shards;
  {p2_causal["audit"]["quality_execution_counts"]["executed"]:,} quality forwards were
  executed and {p2_causal["audit"]["quality_execution_counts"]["reused_exact_config"]:,}
  arm-batches reused an exact byte-identical config; the calibrated+pins versus
  fixed+pins gate passed: **{causal["passed"]}**.
- P2 independent inference: each scale-budget cell has
  {p2_causal["audit"]["independent_seed_clusters_per_cell"]} independent training-seed
  clusters. Exact enumeration covers
  {1 << p2_causal["audit"]["independent_seed_clusters_per_cell"]} sign assignments, so the
  minimum attainable two-sided seed-level p-value is
  {p2_causal["audit"]["minimum_attainable_two_sided_seed_p"]:.4f}. These p-values are
  resolution-limited descriptive evidence and are not used as a p<0.05 success gate;
  the much larger within-seed example count does not increase the number of independently
  trained models.
- P2 supplemental baselines: fixed top-p 0.5/0.8 are evaluated on the complete
  factorial, and the target-aware registered-arm oracle is reported only as a
  non-causal upper bound over {len(p2_causal["offline_oracle_upper_bound"]["registered_arms"])} arms.
- P3 RULER: {p3_ruler["audit"]["completed_cells"]} cells and
  {p3_ruler["audit"]["total_predictions"]:,} predictions on one pinned compatible model.
- P3 natural suite: {p3_natural["audit"]["benchmarks_terminal"]} terminal benchmarks and
  at least {p3_natural["audit"]["minimum_protocol_examples_accounted_per_arm"]:,}
  examples accounted per required arm. The fixed arm was selected before any Qwen3-4B
  outcome as the best of four frozen Qwen3-1.7B RULER candidates at 50% KV over
  8K/16K/32K, then transferred unchanged. "Strongest" is restricted to that selection
  grid and is not a claim of global dominance on Qwen3-4B or every natural benchmark.
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
being imputed. GitHub Actions is currently disabled manually and is not reported as passed;
a successful final CI run remains mandatory before goal completion, and package generation
does not waive it.
"""


def build_package(manifest_path: Path, output_root: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    _require(
        manifest.get("experiment_id") == "adaptive-v4-memory-p5-paper-package-v1",
        "Wrong P5 package manifest.",
    )
    loaded: dict[str, dict[str, Any]] = {}
    inputs: list[dict[str, Any]] = []
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
    )
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
        loaded["p2_core"], loaded["p2_causal"]
    )
    _write_csv(
        output_root / "table-p2-inference-resolution.csv",
        inference_resolution,
        list(inference_resolution[0]),
    )
    quality = _p2_quality_rows(loaded["p2_core"])
    _write_csv(output_root / "table-p2-quality-gate.csv", quality, list(quality[0]))
    p2_core_tables = {
        "table-p2-core-policy-summary.csv": [
            _flatten_json_row(row) for row in loaded["p2_core"]["policy_summary"]
        ],
        "table-p2-core-effects.csv": _p2_core_effect_rows(loaded["p2_core"]),
        "table-p2-core-family-effects.csv": _p2_core_family_rows(loaded["p2_core"]),
        "table-p2-core-seed-effects.csv": _p2_core_seed_rows(loaded["p2_core"]),
        "table-p2-core-worst-slices.csv": _p2_core_worst_slice_rows(loaded["p2_core"]),
    }
    for name, rows in p2_core_tables.items():
        _write_csv(output_root / name, rows, _field_union(rows))
    learned = _learned_lookahead_rows(loaded["p1_online_learned_lookahead"])
    _write_csv(
        output_root / "table-p1-online-learned-lookahead-gate.csv",
        learned,
        list(learned[0]),
    )
    causal = _causal_rows(loaded["p2_causal"])
    _write_csv(output_root / "table-p2-causal-gate.csv", causal, list(causal[0]))
    p2_causal_tables = {
        "table-p2-causal-contrasts.csv": _causal_contrast_rows(loaded["p2_causal"]),
        "table-p2-causal-family-effects.csv": _causal_family_rows(loaded["p2_causal"]),
        "table-p2-causal-seed-effects.csv": _causal_seed_rows(loaded["p2_causal"]),
        "table-p2-causal-worst-slices.csv": _causal_worst_slice_rows(loaded["p2_causal"]),
        "table-p2-causal-physical-memory.csv": _causal_physical_memory_rows(loaded["p2_causal"]),
        "table-p2-causal-offline-oracle.csv": _causal_oracle_rows(loaded["p2_causal"]),
    }
    for name, rows in p2_causal_tables.items():
        _write_csv(output_root / name, rows, _field_union(rows))
    p3 = _p3_rows(loaded["p3_ruler"])
    _write_csv(output_root / "table-p3-ruler-cells.csv", p3, list(p3[0]))
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
    _write_p2_causal_figure(output_root / "figure-p2-causal-effect.svg", loaded["p2_causal"])
    _write_p3_natural_figure(output_root / "figure-p3-natural-quality.svg", loaded["p3_natural"])
    _write_p4_tradeoff_figure(
        output_root / "figure-p4-production-tradeoffs.svg",
        loaded["p4_production_systems"],
    )
    report = _report(
        classifications=classes,
        traceability_rows=traceability_rows,
        p2_core=loaded["p2_core"],
        m5_one_token_pilot=loaded["m5_one_token_pilot"],
        m3_offline_learned_risk_pilot=loaded["m3_offline_learned_risk_pilot"],
        p1_online_learned_lookahead=loaded["p1_online_learned_lookahead"],
        p2_causal=loaded["p2_causal"],
        p3_ruler=loaded["p3_ruler"],
        p3_natural=loaded["p3_natural"],
        p3_safety=loaded["p3_safety"],
        p3_natural_safety=loaded["p3_natural_safety"],
        p3_ifeval=loaded["p3_ifeval"],
        p3_longsafety=loaded["p3_longsafety"],
        p4_500k_context=loaded["p4_500k_context"],
        p4_reference_systems=loaded["p4_reference_systems"],
        p4_production_systems=loaded["p4_production_systems"],
        inputs=inputs,
    )
    (output_root / "paper-report.md").write_text(report)
    generated = [
        output_root / name for name in manifest["generated_files"] if name != "artifact-index.json"
    ]
    commit, dirty = _clean_source()
    _require(not dirty, "P5 package generation requires a clean source tree.")
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
