from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_direct_attestation as attestation  # noqa: E402
import p2_direct_controller_contract as contract  # noqa: E402
import summarize_p2_direct_controller as summary  # noqa: E402

TRUST_ROOT = attestation.TrustRoot(
    key=bytes(range(32)),
    key_id=attestation.derive_key_id(bytes(range(32))),
)
WRONG_TRUST_ROOT = attestation.TrustRoot(
    key=bytes(range(32, 64)),
    key_id=attestation.derive_key_id(bytes(range(32, 64))),
)


def test_contrast_registry_is_fixed_complete_and_nonselective() -> None:
    assert len(summary.ALL_CONTRASTS) == len(summary.CONTRAST_BY_NAME)
    assert tuple(item.comparator for item in summary.CONFIRMATORY_CONTRASTS) == (
        contract.CONVENTIONAL_FIXED_COMPARATOR_ARM,
        contract.CLEAN_ALLOCATOR_CONTROL_ARM,
    )
    assert summary.ORIGINAL_CENTRAL_CAUSAL_CONTRASTS[0].candidate == "calibrated+pins"
    assert summary.ORIGINAL_CENTRAL_CAUSAL_CONTRASTS[0].comparator == "fixed+pins"
    assert all(item.candidate in contract.ALL_ARM_NAMES for item in summary.ALL_CONTRASTS)
    assert all(item.comparator in contract.ALL_ARM_NAMES for item in summary.ALL_CONTRASTS)
    assert {item.comparator for item in summary.TOP_P_SENSITIVITY_CONTRASTS} == set(
        contract.SENSITIVITY_COMPARATOR_ARMS
    )
    registry = summary._analysis_registry()
    source = dict(registry)
    digest = source.pop("registry_sha256")
    assert digest == attestation.checksum(source)
    statistical = registry["manifest_statistical_analysis"]
    gate = registry["manifest_confirmatory_success_gate"]
    assert statistical["confirmatory_comparators_pooled_selected_or_dropped"] is False
    assert statistical["top_p_sensitivity_in_confirmatory_multiplicity"] is False
    assert gate["top_p_eligible_for_primary_gate"] is False


def test_seed_cluster_inference_exposes_five_seed_resolution() -> None:
    first = summary.seed_cluster_statistics(
        [0.01, 0.02, 0.03, 0.04, 0.05],
        label="resolution",
        resamples=1_000,
    )
    second = summary.seed_cluster_statistics(
        [0.01, 0.02, 0.03, 0.04, 0.05],
        label="resolution",
        resamples=1_000,
    )
    assert first == second
    assert first["exact_seed_sign_flip_assignments"] == 32
    assert first["minimum_attainable_exact_two_sided_p"] == pytest.approx(0.0625)
    assert first["exact_seed_sign_flip_two_sided_p"] == pytest.approx(0.0625)
    assert first["minimum_p_exceeds_nominal_alpha"] is True
    assert first["seed_cluster_bootstrap_ci"][0] > 0.0


def test_paired_conversation_bootstrap_is_deterministic_and_conditional() -> None:
    values = [0.25] * 80 + [-0.25] * 20
    first = summary.paired_conversation_statistics(
        values,
        label="paired",
        resamples=1_000,
    )
    second = summary.paired_conversation_statistics(
        values,
        label="paired",
        resamples=1_000,
    )
    assert first == second
    assert first["paired_conversations"] == 100
    assert first["mean_difference"] == pytest.approx(0.15)
    assert first["population_generalization_claimed"] is False


def test_large_paired_support_uses_memory_bounded_resampling_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    support_cardinality = 2_001
    resamples = 1_000
    requested_shapes: list[tuple[str, int, int]] = []

    class ShapeSpyGenerator:
        def multinomial(
            self,
            observation_count: int,
            probabilities: object,
            *,
            size: int,
        ) -> object:
            probability_array = summary.np.asarray(probabilities)
            requested_shapes.append(("multinomial", size, len(probability_array)))
            result = summary.np.zeros((size, len(probability_array)), dtype=summary.np.int64)
            result[:, 0] = observation_count
            return result

        def binomial(
            self,
            counts: object,
            _probability: float,
            *,
            size: tuple[int, int],
        ) -> object:
            count_array = summary.np.asarray(counts)
            assert size[1] == len(count_array)
            requested_shapes.append(("binomial", size[0], size[1]))
            return summary.np.zeros(size, dtype=summary.np.int64)

    monkeypatch.setattr(summary, "_numpy_rng", lambda _label: ShapeSpyGenerator())
    statistics = summary._paired_conversation_statistics_from_support(
        summary.np.linspace(-1.0, 1.0, support_cardinality),
        [1] * support_cardinality,
        label="large-support",
        resamples=resamples,
    )
    assert statistics["paired_conversations"] == support_cardinality
    assert len(requested_shapes) > 2
    assert all(
        rows * columns <= summary.MAX_RESAMPLE_SUPPORT_MATRIX_ENTRIES
        for _kind, rows, columns in requested_shapes
    )
    assert all(
        (rows, columns) != (resamples, support_cardinality)
        for _kind, rows, columns in requested_shapes
    )


def _token_row() -> dict[str, object]:
    materialization = [
        {
            "layer_index": layer,
            "capacity_blocks": 1,
            "hot_blocks": 1,
            "hot_bytes": 16,
        }
        for layer in contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    ]
    diagnostics = [
        {
            "clip_saturated": layer == 2,
            "history_regime": "steady-state",
            "applied_quota_blocks": 1,
            "next_allocated_quota_blocks": 2,
        }
        for layer in contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
    ]
    return {
        "applied_materialization": materialization,
        "decode_incremental_transfer_deltas": [
            {"h2d_delta_bytes": 8, "d2h_delta_bytes": 4}
            for _ in contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
        ],
        "cuda_peak_allocated_bytes": 128,
        "cuda_peak_reserved_bytes": 256,
        "is_cuda_hbm_evidence": True,
        "runtime_soft_lag_snapshot": {
            "cuda_peak_allocated_bytes": 96,
            "cuda_peak_reserved_bytes": 192,
            "is_cuda_hbm_evidence": True,
        },
        "signal_diagnostics": diagnostics,
    }


def test_system_accumulator_uses_common_peaks_and_physical_bytes() -> None:
    accumulator = summary.ArmSliceAccumulator()
    accumulator.add_outcome(
        {
            "status": "success",
            "accuracy": 0.75,
            "wall_time_ns": 100,
            "tokens_per_second": 10.0,
            "decoded_tokens": 1,
        }
    )
    assert accumulator.add_token(_token_row(), exact_fill=True) == 48
    payload = accumulator.payload()
    assert payload["hot_resident_bytes"]["mean"] == 48
    assert payload["h2d_delta_bytes"]["sum"] == 24
    assert payload["d2h_delta_bytes"]["sum"] == 12
    assert payload["cuda_peak_allocated_bytes"]["mean"] == 128
    assert payload["exact_fill_violations"] == 0
    assert payload["clip_saturation_rate"] == pytest.approx(1 / 3)
    assert payload["quota_movement_rate"] == 1.0
    assert payload["signal_diagnostics_by_regime"] == [
        {
            "regime": "steady-state",
            "signal_rows": 3,
            "clip_saturated_rows": 1,
            "clip_saturation_rate": pytest.approx(1 / 3),
            "quota_movement_eligible_rows": 3,
            "quota_movement_rows": 3,
            "quota_movement_rate": 1.0,
        }
    ]


def test_system_accumulator_rejects_legacy_transfer_field() -> None:
    row = _token_row()
    deltas = row.pop("decode_incremental_transfer_deltas")
    row["static_transfer_deltas"] = deltas
    with pytest.raises(ValueError, match="Token transfer deltas are missing"):
        summary.ArmSliceAccumulator().add_token(row, exact_fill=True)


def test_collect_validated_study_consumes_each_raw_row_in_one_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "EXAMPLES_PER_SHARD", 1)
    training_seed = contract.TRAINING_SEEDS[0]
    _, calibration_seed, evaluation_seed = contract.seed_triplet(training_seed)
    record: dict[str, Any] = {
        "scale": "s55",
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
        "budget": "2x",
        "family": contract.FAMILIES[0],
        "context": contract.CONTEXTS[0],
        "replicate": contract.REPLICATES[0],
        "generation_seed": contract.generation_seed(
            evaluation_seed,
            contract.FAMILIES[0],
            contract.CONTEXTS[0],
            contract.REPLICATES[0],
        ),
    }
    coordinate = summary._coordinate_from_record(record)
    execution_order = summary.evaluator.arm_execution_order(
        summary.evaluator.shard_schedule_index(
            family=record["family"],
            context=record["context"],
            replicate=record["replicate"],
            example_index=0,
        )
    )
    outcomes = [
        {
            "example_index": 0,
            "arm": arm,
            "status": "success",
            "accuracy": 1.0,
            "all_queries_correct": True,
            "correct_count": 1,
            "total": 1,
            "wall_time_ns": 100,
            "tokens_per_second": 10.0,
            "decoded_tokens": 1,
        }
        for arm in execution_order
    ]
    tokens = []
    for arm in execution_order:
        token = _token_row()
        token.update(
            {
                "example_index": 0,
                "arm": arm,
                "token_index": 0,
                "post_rebalance_materialization": [
                    {"h2d_bytes": 10, "d2h_bytes": 5}
                    for _ in contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"]
                ],
            }
        )
        tokens.append(token)
    envelope = {
        "coordinate": coordinate,
        "inputs": {
            "calibration_artifact": {"sha256": "a" * 64},
            "top_p_match_artifacts": {
                arm: {"sha256": str(index) * 64}
                for index, arm in enumerate(contract.SENSITIVITY_COMPARATOR_ARMS, start=1)
            },
        },
    }

    def fake_iterator(
        _payload: object,
        **callbacks: object,
    ) -> object:
        on_outcome = callbacks["outcome_callback"]
        on_token = callbacks["token_callback"]
        assert callable(on_outcome) and callable(on_token)
        for row in outcomes:
            on_outcome(record, row)
        for row in tokens:
            on_token(record, row)
        yield record, envelope

    monkeypatch.setattr(summary.integrity_audit, "iter_validated_raw_shards", fake_iterator)
    monkeypatch.setattr(summary, "validate_study_coverage", lambda _accumulator: {})
    monkeypatch.setattr(
        summary.evaluator,
        "iter_direct_controller_sidecar_rows",
        lambda *_args, **_kwargs: pytest.fail("summary reopened a validated sidecar"),
    )
    accumulator = summary.collect_validated_study({}, trust_root=TRUST_ROOT)
    assert accumulator.raw_outcome_rows == len(contract.ALL_ARM_NAMES)
    assert accumulator.raw_token_rows == len(contract.ALL_ARM_NAMES)
    assert accumulator.raw_failure_rows == 0
    assert all(
        item.runs_with_terminal_transfer_evidence == 1
        and item.total_h2d_bytes_including_initial_tiering.total == 30
        for item in accumulator.arm_slices.values()
    )


_VALID_UNSIGNED_TEMPLATE: dict[str, object] | None = None


def _minimal_unsigned_summary() -> dict[str, object]:
    global _VALID_UNSIGNED_TEMPLATE
    if _VALID_UNSIGNED_TEMPLATE is not None:
        return copy.deepcopy(_VALID_UNSIGNED_TEMPLATE)

    def effect_fields(paired_conversations: int) -> dict[str, object]:
        return {
            "paired_conversations": paired_conversations,
            "mean_difference": 0.1,
            "mean_difference_percentage_points": 10.0,
            "candidate_mean": 0.6,
            "comparator_mean": 0.5,
            "absolute_mean_difference_replayed": 0.1,
            "relative_mean_difference": 0.2,
            "relative_mean_difference_percent": 20.0,
            "relative_effect_defined": True,
        }

    def paired_fields(count: int, label: str) -> dict[str, object]:
        return summary._paired_conversation_statistics_from_support(
            [0.1],
            [count],
            label=label,
        )

    def seed_inference(label: str) -> dict[str, object]:
        return summary.seed_cluster_statistics(
            [0.1 for _seed in contract.TRAINING_SEEDS],
            label=label,
        )

    conversations_per_seed = (
        len(contract.FAMILIES)
        * len(contract.CONTEXTS)
        * len(contract.REPLICATES)
        * contract.EXAMPLES_PER_SHARD
    )

    def contrast_row(contrast: summary.Contrast, metric: str) -> dict[str, object]:
        seed_rows = [
            {
                "scale": scale,
                "budget": budget,
                "training_seed": seed,
                **effect_fields(conversations_per_seed),
                "candidate_technical_failures": 0,
                "comparator_technical_failures": 0,
            }
            for scale in contract.SCALES
            for budget in contract.BUDGETS
            for seed in contract.TRAINING_SEEDS
        ]
        cells = [
            {
                "scale": scale,
                "budget": budget,
                **effect_fields(conversations_per_seed * len(contract.TRAINING_SEEDS)),
                **paired_fields(
                    conversations_per_seed * len(contract.TRAINING_SEEDS),
                    summary._paired_cell_label(
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                    ),
                ),
                "seed_cluster_inference": seed_inference(
                    summary._seed_cell_label(
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                    )
                ),
                "candidate_technical_failures": 0,
                "comparator_technical_failures": 0,
                "all_five_seed_effects_positive": True,
                "all_five_seed_effects_nonnegative": True,
                "holm_adjusted_seed_exact_p_across_four_cells": 0.25,
                "holm_cell_family_size": 4,
                **(
                    {
                        "holm_adjusted_seed_exact_p_across_diagnostic_contrasts": 0.9375,
                        "holm_diagnostic_contrast_family_size": len(
                            summary.CAUSAL_DIAGNOSTIC_CONTRASTS
                        ),
                    }
                    if contrast in summary.CAUSAL_DIAGNOSTIC_CONTRASTS
                    else {}
                ),
            }
            for scale in contract.SCALES
            for budget in contract.BUDGETS
        ]
        family_rows = [
            {
                "scale": scale,
                "budget": budget,
                "family": family,
                **effect_fields(
                    len(contract.TRAINING_SEEDS)
                    * len(contract.CONTEXTS)
                    * len(contract.REPLICATES)
                    * contract.EXAMPLES_PER_SHARD
                ),
                **paired_fields(
                    len(contract.TRAINING_SEEDS)
                    * len(contract.CONTEXTS)
                    * len(contract.REPLICATES)
                    * contract.EXAMPLES_PER_SHARD,
                    summary._paired_family_label(
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                        family=family,
                    ),
                ),
                "seed_cluster_inference": seed_inference(
                    summary._seed_family_label(
                        metric=metric,
                        contrast=contrast.name,
                        scale=scale,
                        budget=budget,
                        family=family,
                    )
                ),
                "by_training_seed": [
                    {
                        "training_seed": seed,
                        **effect_fields(
                            len(contract.CONTEXTS)
                            * len(contract.REPLICATES)
                            * contract.EXAMPLES_PER_SHARD
                        ),
                    }
                    for seed in contract.TRAINING_SEEDS
                ],
                "holm_adjusted_seed_exact_p_across_families": 0.5625,
                "holm_family_size": len(contract.FAMILIES),
            }
            for scale in contract.SCALES
            for budget in contract.BUDGETS
            for family in contract.FAMILIES
        ]
        context_rows = [
            {
                "scale": scale,
                "budget": budget,
                "context": context,
                **effect_fields(
                    len(contract.TRAINING_SEEDS)
                    * len(contract.FAMILIES)
                    * len(contract.REPLICATES)
                    * contract.EXAMPLES_PER_SHARD
                ),
            }
            for scale in contract.SCALES
            for budget in contract.BUDGETS
            for context in contract.CONTEXTS
        ]
        slice_rows = [
            {
                "scale": scale,
                "budget": budget,
                "family": family,
                "context": context,
                **effect_fields(
                    len(contract.TRAINING_SEEDS)
                    * len(contract.REPLICATES)
                    * contract.EXAMPLES_PER_SHARD
                ),
            }
            for scale in contract.SCALES
            for budget in contract.BUDGETS
            for family in contract.FAMILIES
            for context in contract.CONTEXTS
        ]
        joint_slice_rows = [
            {
                "scale": scale,
                "budget": budget,
                "training_seed": seed,
                "family": family,
                "context": context,
                **effect_fields(len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD),
            }
            for scale in contract.SCALES
            for budget in contract.BUDGETS
            for family in contract.FAMILIES
            for context in contract.CONTEXTS
            for seed in contract.TRAINING_SEEDS
        ]
        return {
            "name": contrast.name,
            "candidate": contrast.candidate,
            "comparator": contrast.comparator,
            "role": contrast.role,
            "interpretation": contrast.interpretation,
            "metric": metric,
            "failure_handling": {
                "estimand": "intent-to-treat",
                "technical_failure_score": summary.FAILURE_SCORE,
                "technical_failures_retained": True,
                "confirmatory_success_requires_zero_failures": contrast
                in summary.PHYSICAL_CONFIRMATORY_CONTRASTS,
            },
            "cells": cells,
            "by_seed": seed_rows,
            "worst_seed": seed_rows[0],
            "worst_seed_by_scale_budget": summary._worst_by_scale_budget(seed_rows),
            "by_family_with_holm_bonferroni": family_rows,
            "by_context": context_rows,
            "worst_context": context_rows[0],
            "worst_context_by_scale_budget": summary._worst_by_scale_budget(context_rows),
            "by_family_context": slice_rows,
            "worst_slice": slice_rows[0],
            "worst_slice_by_scale_budget": summary._worst_by_scale_budget(slice_rows),
            "by_training_seed_family_context": joint_slice_rows,
            "worst_joint_slice": joint_slice_rows[0],
            "worst_joint_slice_by_scale_budget": summary._worst_by_scale_budget(joint_slice_rows),
        }

    diagnostic_multiplicity = {
        "method": "Holm-Bonferroni",
        "source_p": "exact independent-training-seed sign-flip two-sided p",
        "families": "one family per scale-budget cell",
        "contrasts_per_family": len(summary.CAUSAL_DIAGNOSTIC_CONTRASTS),
        "family_cells": len(contract.SCALES) * len(contract.BUDGETS),
        "top_p_sensitivity_included": False,
        "confirmatory_comparators_pooled": False,
    }
    quality = {
        metric: {
            "original_central_causal": {
                item.name: contrast_row(item, metric)
                for item in summary.ORIGINAL_CENTRAL_CAUSAL_CONTRASTS
            },
            "confirmatory": {
                item.name: contrast_row(item, metric) for item in summary.CONFIRMATORY_CONTRASTS
            },
            "causal_diagnostics": {
                item.name: contrast_row(item, metric)
                for item in summary.CAUSAL_DIAGNOSTIC_CONTRASTS
            },
            "top_p_descriptive_sensitivity": {
                item.name: contrast_row(item, metric)
                for item in summary.TOP_P_SENSITIVITY_CONTRASTS
            },
            "diagnostic_multiplicity": diagnostic_multiplicity,
        }
        for metric in summary.QUALITY_METRICS
    }

    def system_count_fields(
        *,
        arm: str,
        outcomes: int,
        token_rows: int,
        signal_rows: int,
    ) -> dict[str, object]:
        def moments(count: int, total: float) -> dict[str, object]:
            mean = total / count
            return {
                "count": count,
                "sum": total,
                "sum_squared": count * mean * mean,
                "mean": mean,
                "sample_standard_deviation": 0.0,
                "minimum": mean,
                "maximum": mean,
            }

        return {
            "outcomes": outcomes,
            "successes": outcomes,
            "technical_failures": 0,
            "intent_to_treat_accuracy": moments(outcomes, float(outcomes)),
            "wall_time_ns": moments(outcomes, float(outcomes)),
            "tokens_per_second": moments(outcomes, float(outcomes)),
            "decoded_tokens": moments(outcomes, float(token_rows)),
            "token_rows": token_rows,
            "hot_resident_bytes": moments(token_rows, float(token_rows)),
            "h2d_delta_bytes": moments(token_rows, 0.0),
            "d2h_delta_bytes": moments(token_rows, 0.0),
            "decode_incremental_transfer_scope": (
                "token deltas exclude initial enable_csa_tiering counters"
            ),
            "runs_with_terminal_transfer_evidence": outcomes,
            "total_h2d_bytes_including_initial_tiering_per_run": moments(outcomes, 0.0),
            "total_d2h_bytes_including_initial_tiering_per_run": moments(outcomes, 0.0),
            "missing_cuda_peak_rows": 0,
            "cuda_hbm_evidence_rows": token_rows,
            "cuda_peak_allocated_bytes": moments(token_rows, float(token_rows)),
            "cuda_peak_reserved_bytes": moments(token_rows, float(token_rows * 2)),
            "all_token_rows_have_cuda_hbm_evidence": True,
            "exact_fill_rows": token_rows if arm in contract.EXACT_FILL_ARM_NAMES else 0,
            "exact_fill_violations": 0,
            "signal_rows": signal_rows,
            "clip_saturated_rows": 0,
            "clip_saturation_rate": 0.0,
            "quota_movement_eligible_rows": 0,
            "quota_movement_rows": 0,
            "quota_movement_rate": None,
            "signal_regimes": {"steady-state": signal_rows},
            "signal_diagnostics_by_regime": [
                {
                    "regime": "steady-state",
                    "signal_rows": signal_rows,
                    "clip_saturated_rows": 0,
                    "clip_saturation_rate": 0.0,
                    "quota_movement_eligible_rows": 0,
                    "quota_movement_rows": 0,
                    "quota_movement_rate": None,
                }
            ],
        }

    expected_cells = sorted(
        (scale, budget, seed, family, context)
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
    )
    system_slices = []
    aggregate_by_arm: dict[str, dict[str, int]] = {}
    for arm in contract.ALL_ARM_NAMES:
        arm_tokens = 0
        arm_signals = 0
        for scale, budget, seed, family, context in expected_cells:
            token_rows = (
                contract.DECODE_TOKENS_PER_EXAMPLE_BY_FAMILY_CONTEXT[family][context]
                * len(contract.REPLICATES)
                * contract.EXAMPLES_PER_SHARD
            )
            signal_rows = token_rows * len(contract.DIRECT_CSA_LAYERS_BY_SCALE[scale])
            arm_tokens += token_rows
            arm_signals += signal_rows
            system_slices.append(
                {
                    "arm": arm,
                    "scale": scale,
                    "budget": budget,
                    "training_seed": seed,
                    "family": family,
                    "context": context,
                    **system_count_fields(
                        arm=arm,
                        outcomes=len(contract.REPLICATES) * contract.EXAMPLES_PER_SHARD,
                        token_rows=token_rows,
                        signal_rows=signal_rows,
                    ),
                }
            )
        aggregate_by_arm[arm] = {"token_rows": arm_tokens, "signal_rows": arm_signals}

    per_arm = [
        {
            "arm": arm,
            **system_count_fields(
                arm=arm,
                outcomes=contract.BUDGET_SHARDS_TOTAL * contract.EXAMPLES_PER_SHARD,
                token_rows=aggregate_by_arm[arm]["token_rows"],
                signal_rows=aggregate_by_arm[arm]["signal_rows"],
            ),
        }
        for arm in contract.ALL_ARM_NAMES
    ]
    pair_ids = sorted(
        (contrast.name, scale, budget, seed)
        for contrast in summary.PHYSICAL_CONFIRMATORY_CONTRASTS
        for scale in contract.SCALES
        for budget in contract.BUDGETS
        for seed in contract.TRAINING_SEEDS
    )
    physical_pairs = [
        {
            "contrast": contrast_name,
            "candidate": summary.CONTRAST_BY_NAME[contrast_name].candidate,
            "comparator": summary.CONTRAST_BY_NAME[contrast_name].comparator,
            "scale": scale,
            "budget": budget,
            "training_seed": seed,
            "paired_token_rows": 406_000,
            "exact_byte_matches": 406_000,
            "exact_byte_match_rate": 1.0,
            "absolute_difference_bytes": 0,
            "maximum_absolute_difference_bytes": 0,
            "missing_candidate_rows": 0,
            "missing_comparator_rows": 0,
            "exact_physical_hot_byte_parity": True,
        }
        for contrast_name, scale, budget, seed in pair_ids
    ]
    top_p_records = [
        {
            "scale": scale,
            "training_seed": seed,
            "calibration_seed": contract.seed_triplet(seed)[1],
            "budget": budget,
            "comparator": comparator,
            "terminal_decision": "GO",
            "observation_count": contract.TOP_P_MATCH_OBSERVATION_COUNT,
            "target_hot_resident_bytes_total": 45_000,
            "comparator_hot_resident_bytes_total": 45_000,
            "relative_difference": 0.0,
            "artifact_binding": {
                "path": f"/{scale}/{seed}/{budget}/{comparator}.json",
                "bytes": 1,
                "sha256": "1" * 64,
                "payload_sha256": "2" * 64,
                "attestation_mac": "3" * 64,
                "experiment_id": "top-p-test",
            },
        }
        for scale, seed, budget, comparator in sorted(
            (scale, seed, budget, comparator)
            for scale in contract.SCALES
            for seed in contract.TRAINING_SEEDS
            for budget in contract.BUDGETS
            for comparator in contract.SENSITIVITY_COMPARATOR_ARMS
        )
    ]
    system = {
        "scope": "P2 reference sequential-tiered evaluator; production P4 latency claim excluded",
        "whole_process_hbm_claimed": False,
        "physical_hot_tensor_bytes_claimed": True,
        "decode_incremental_h2d_d2h_reported": True,
        "run_total_h2d_d2h_including_initial_tiering_reported": True,
        "latency_tail_quantiles_claimed": False,
        "per_arm": per_arm,
        "by_seed_scale_budget_family_context": system_slices,
        "confirmatory_paired_hot_byte_parity": physical_pairs,
        "all_confirmatory_pairs_exact_hot_byte_parity": True,
        "all_exact_fill_rows_satisfied": True,
        "all_token_rows_have_cuda_hbm_evidence": True,
        "signal_diagnostics_reported_without_quality_selection": True,
        "technical_failure_accounting": {
            "total_failures": 0,
            "total_timeouts": 0,
            "failure_types": [],
            "by_arm_seed_scale_budget_family_context_and_type": [],
            "failures_retained_as_intent_to_treat_zero": True,
        },
        "calibration_only_top_p_physical_match": {
            "scope": "calibration-only; evaluation seeds and quality outcomes inaccessible",
            "match_metric": contract.PHYSICAL_MATCH_TARGET_METRIC,
            "maximum_relative_difference": contract.MAX_RELATIVE_HOT_BYTES_DIFFERENCE,
            "fixed_mixture_rule": contract.FIXED_MIXTURE_RULE,
            "records": top_p_records,
            "all_terminal_go": True,
            "all_within_one_percent": True,
            "used_for_primary_confirmatory_decision": False,
        },
    }
    primary_quality = quality[summary.PRIMARY_QUALITY_METRIC]
    central = cast(dict[str, dict[str, Any]], primary_quality["original_central_causal"])
    confirmatory = cast(dict[str, dict[str, Any]], primary_quality["confirmatory"])
    replay_contrasts: dict[str, dict[str, Any]] = {**central, **confirmatory}
    decision = summary.decision_summary(
        replay_contrasts,
        system_summary=system,
    )
    implementation_digest = "a" * 64
    result: dict[str, object] = {
        "schema_version": summary.SCHEMA_VERSION,
        "experiment_id": summary.EXPERIMENT_ID,
        "artifact_type": summary.ARTIFACT_TYPE,
        "status": "terminal",
        "source": {
            "integrity_artifact": {
                "path": "/integrity.json",
                "sha256": "4" * 64,
                "bytes": 1,
                "payload_sha256": "5" * 64,
                "attestation_scheme": attestation.SCHEME,
                "attestation_key_id": TRUST_ROOT.key_id,
                "attestation_mac": "6" * 64,
            },
            "frozen_analysis_provenance": {
                "implementation_tree_sha256": implementation_digest,
                "current_source_state": {"commit": "c" * 40, "dirty": False},
                "integrity_source_state": {"commit": "d" * 40, "dirty": False},
                "manifest_binding": {
                    "implementation_digest": implementation_digest,
                    "implementation_source_commit": "b" * 40,
                    "attestation": attestation.public_manifest_contract(TRUST_ROOT.key_id),
                },
                "implementation_digest_matches_manifest": True,
                "current_source_clean": True,
                "integrity_and_implementation_commits_are_ancestors": True,
                "attestation_key_matches_manifest": True,
            },
            "analysis_environment": summary._analysis_environment_binding(),
            "raw_artifacts_revalidated_before_analysis": True,
        },
        "preregistration": summary._preregistration_payload(),
        "coverage": {
            "shards": contract.BUDGET_SHARDS_TOTAL,
            "distinct_generated_conversations": contract.DISTINCT_GENERATED_CONVERSATIONS_TOTAL,
            "scale_specific_conversation_evaluations": (
                contract.SCALE_SPECIFIC_CONVERSATION_EVALUATIONS_TOTAL
            ),
            "budget_expanded_conversation_evaluations": (
                contract.BUDGET_EXPANDED_CONVERSATION_EVALUATIONS_TOTAL
            ),
            "arm_conversation_outcomes": contract.BUDGET_SHARDS_TOTAL
            * contract.EXAMPLES_PER_SHARD
            * len(contract.ALL_ARM_NAMES),
            "paired_units_per_seed_scale_budget_family_context": len(contract.REPLICATES)
            * contract.EXAMPLES_PER_SHARD,
            "paired_units_per_seed_scale_budget_family": contract.EXAMPLES_PER_FAMILY,
            "raw_token_rows": contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES,
            "raw_token_rows_without_technical_failures": (
                contract.EXPECTED_RAW_TOKEN_ROWS_WITHOUT_FAILURES
            ),
            "raw_failure_rows": 0,
            "calibration_artifacts": len(contract.SCALES) * len(contract.TRAINING_SEEDS),
            "top_p_physical_match_artifacts": len(top_p_records),
        },
        "paired_quality_statistics": quality,
        "physical_and_reference_system_statistics": system,
        "decision": decision,
        "analysis_guard": summary._analysis_guard_payload(),
        "claim_boundary": summary._claim_boundary_payload(),
    }
    _VALID_UNSIGNED_TEMPLATE = result
    return copy.deepcopy(result)


def test_summary_hmac_rejects_tampering_and_wrong_key() -> None:
    artifact = summary.seal_summary_payload(
        _minimal_unsigned_summary(),
        trust_root=TRUST_ROOT,
    )
    assert (
        summary.validate_summary_artifact(
            artifact, trust_root=TRUST_ROOT, verify_source_bindings=False
        )
        == artifact
    )
    with pytest.raises(ValueError):
        summary.validate_summary_artifact(
            artifact, trust_root=WRONG_TRUST_ROOT, verify_source_bindings=False
        )
    tampered = copy.deepcopy(artifact)
    tampered["status"] = "in_progress"
    with pytest.raises(ValueError):
        summary.validate_summary_artifact(
            tampered, trust_root=TRUST_ROOT, verify_source_bindings=False
        )


def test_summary_default_validation_rejects_missing_integrity_artifact() -> None:
    artifact = summary.seal_summary_payload(
        _minimal_unsigned_summary(),
        trust_root=TRUST_ROOT,
    )
    with pytest.raises(ValueError, match="could not be opened safely"):
        summary.validate_summary_artifact(artifact, trust_root=TRUST_ROOT)


def _assert_resealed_summary_rejected(
    payload: dict[str, object],
    *,
    match: str,
) -> None:
    artifact = summary.seal_summary_payload(payload, trust_root=TRUST_ROOT)
    with pytest.raises(ValueError, match=match):
        summary.validate_summary_artifact(
            artifact, trust_root=TRUST_ROOT, verify_source_bindings=False
        )


def test_valid_summary_fixture_has_full_frozen_cardinality() -> None:
    payload = _minimal_unsigned_summary()
    coverage = payload["coverage"]
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(coverage, dict) and isinstance(system, dict)
    assert coverage["raw_token_rows_without_technical_failures"] == 154_280_000
    assert coverage["distinct_generated_conversations"] == 45_000
    assert coverage["scale_specific_conversation_evaluations"] == 90_000
    assert coverage["budget_expanded_conversation_evaluations"] == 180_000
    assert len(system["per_arm"]) == 19
    assert len(system["by_seed_scale_budget_family_context"]) == 17_100
    assert len(system["confirmatory_paired_hot_byte_parity"]) == 60
    top_p = system["calibration_only_top_p_physical_match"]
    assert isinstance(top_p, dict) and len(top_p["records"]) == 40
    assert all(
        row["observation_count"] == contract.TOP_P_MATCH_OBSERVATION_COUNT
        for row in top_p["records"]
    )


def test_signed_summary_binds_exact_numpy_analysis_environment() -> None:
    payload = _minimal_unsigned_summary()
    source = cast(dict[str, Any], payload["source"])
    environment = cast(dict[str, Any], source["analysis_environment"])
    assert environment["numpy_rng"] == "numpy.random.Generator(numpy.random.PCG64)"
    assert environment["numpy_quantile_method"] == "linear"
    assert environment["exact_dependency_lock_bound"] is False
    environment["numpy_version"] = "0.0.0-drifted"
    _assert_resealed_summary_rejected(payload, match="analysis environment")


@pytest.mark.parametrize(
    ("section", "field", "match"),
    (
        ("source", "raw_artifacts_revalidated_before_analysis", "source"),
        ("preregistration", "failure_policy", "preregistration"),
        ("analysis_guard", "calibration_seed_used_for_quality_generation", "guard"),
        ("claim_boundary", "whole_process_hbm_claimed", "claim boundary"),
        ("coverage", "raw_token_rows_without_technical_failures", "coverage"),
    ),
)
def test_signed_summary_rejects_semantic_section_omission(
    section: str,
    field: str,
    match: str,
) -> None:
    payload = _minimal_unsigned_summary()
    value = payload[section]
    assert isinstance(value, dict)
    value.pop(field)
    _assert_resealed_summary_rejected(payload, match=match)


@pytest.mark.parametrize(
    ("collection", "match"),
    (
        ("per_arm", "Per-arm system inventory"),
        ("by_seed_scale_budget_family_context", "arm-slice grid"),
        ("confirmatory_paired_hot_byte_parity", "physical-pair"),
    ),
)
def test_signed_summary_rejects_system_cardinality_omission(
    collection: str,
    match: str,
) -> None:
    payload = _minimal_unsigned_summary()
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(system, dict)
    rows = system[collection]
    assert isinstance(rows, list)
    rows.pop()
    _assert_resealed_summary_rejected(payload, match=match)


def test_signed_summary_rejects_top_p_cardinality_omission() -> None:
    payload = _minimal_unsigned_summary()
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(system, dict)
    top_p = system["calibration_only_top_p_physical_match"]
    assert isinstance(top_p, dict) and isinstance(top_p["records"], list)
    top_p["records"].pop()
    _assert_resealed_summary_rejected(payload, match="Top-p physical-match summary replay")


@pytest.mark.parametrize(
    ("collection", "match"),
    (
        ("cells", "slice cardinality"),
        ("by_seed", "slice cardinality"),
        ("by_family_with_holm_bonferroni", "slice cardinality"),
        ("by_context", "slice cardinality"),
        ("by_family_context", "slice cardinality"),
        ("by_training_seed_family_context", "slice cardinality"),
    ),
)
def test_signed_summary_rejects_contrast_cardinality_omission(
    collection: str,
    match: str,
) -> None:
    payload = _minimal_unsigned_summary()
    quality = payload["paired_quality_statistics"]
    assert isinstance(quality, dict)
    primary = quality[summary.PRIMARY_QUALITY_METRIC]
    assert isinstance(primary, dict)
    central = primary["original_central_causal"]
    assert isinstance(central, dict)
    contrast = central[contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME]
    assert isinstance(contrast, dict)
    rows = contrast[collection]
    assert isinstance(rows, list)
    rows.pop()
    _assert_resealed_summary_rejected(payload, match=match)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("fixed_coverage", "fixed coverage"),
        ("dynamic_tokens", "aggregates disagree"),
        ("failure_total", "aggregates disagree"),
        ("physical_pair_tokens", "Zero-failure confirmatory physical pairing"),
    ),
)
def test_signed_summary_rejects_cardinality_mutation(mutation: str, match: str) -> None:
    payload = _minimal_unsigned_summary()
    coverage = payload["coverage"]
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(coverage, dict) and isinstance(system, dict)
    if mutation == "fixed_coverage":
        coverage["arm_conversation_outcomes"] = int(coverage["arm_conversation_outcomes"]) - 1
    elif mutation == "dynamic_tokens":
        coverage["raw_token_rows"] = int(coverage["raw_token_rows"]) - 1
    elif mutation == "failure_total":
        coverage["raw_failure_rows"] = 1
    else:
        pairs = system["confirmatory_paired_hot_byte_parity"]
        assert isinstance(pairs, list) and isinstance(pairs[0], dict)
        pairs[0]["paired_token_rows"] = 405_999
        pairs[0]["exact_byte_matches"] = 405_999
    _assert_resealed_summary_rejected(payload, match=match)


@pytest.mark.parametrize(
    ("section", "field", "replacement", "match"),
    (
        ("paired", "paired_bootstrap_ci", [-0.2, 0.2], "paired-statistic replay"),
        (
            "paired",
            "paired_monte_carlo_sign_flip_two_sided_p",
            0.5,
            "paired-statistic replay",
        ),
        ("seed", "seed_cluster_bootstrap_ci", [-0.2, 0.2], "seed-cluster inference"),
        ("seed", "exact_seed_sign_flip_two_sided_p", 0.5, "seed-cluster inference"),
    ),
)
def test_signed_summary_recomputes_frozen_inference(
    section: str,
    field: str,
    replacement: object,
    match: str,
) -> None:
    payload = _minimal_unsigned_summary()
    quality = payload["paired_quality_statistics"]
    assert isinstance(quality, dict)
    primary = quality[summary.PRIMARY_QUALITY_METRIC]
    assert isinstance(primary, dict)
    central = primary["original_central_causal"]
    assert isinstance(central, dict)
    contrast = central[contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME]
    assert isinstance(contrast, dict)
    cells = contrast["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    target = cells[0]
    if section == "seed":
        seed = target["seed_cluster_inference"]
        assert isinstance(seed, dict)
        target = seed
    target[field] = replacement
    _assert_resealed_summary_rejected(payload, match=match)


def test_signed_summary_rejects_altered_same_mean_family_seed_vector() -> None:
    payload = _minimal_unsigned_summary()
    quality = cast(dict[str, Any], payload["paired_quality_statistics"])
    primary = cast(dict[str, Any], quality[summary.PRIMARY_QUALITY_METRIC])
    central = cast(dict[str, Any], primary["original_central_causal"])
    contrast = cast(
        dict[str, Any],
        central[contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME],
    )
    family = cast(list[dict[str, Any]], contrast["by_family_with_holm_bonferroni"])[0]
    seed_rows = cast(list[dict[str, Any]], family["by_training_seed"])
    for seed_row, candidate, difference in (
        (seed_rows[0], 0.55, 0.05),
        (seed_rows[1], 0.65, 0.15),
    ):
        seed_row.update(
            {
                "candidate_mean": candidate,
                "mean_difference": difference,
                "mean_difference_percentage_points": difference * 100.0,
                "absolute_mean_difference_replayed": difference,
                "relative_mean_difference": difference / 0.5,
                "relative_mean_difference_percent": difference / 0.5 * 100.0,
            }
        )
    family["seed_cluster_inference"] = summary.seed_cluster_statistics(
        [float(row["mean_difference"]) for row in seed_rows],
        label=summary._seed_family_label(
            metric=summary.PRIMARY_QUALITY_METRIC,
            contrast=contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME,
            scale=str(family["scale"]),
            budget=str(family["budget"]),
            family=str(family["family"]),
        ),
    )
    _assert_resealed_summary_rejected(
        payload,
        match="family seed from training-seed-family-context rows",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("observation_count", contract.TOP_P_MATCH_OBSERVATION_COUNT - 1),
        ("comparator_hot_resident_bytes_total", 45_001),
    ),
)
def test_signed_summary_rejects_top_p_arithmetic_inconsistency(
    field: str,
    value: int,
) -> None:
    payload = _minimal_unsigned_summary()
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(system, dict)
    top_p = system["calibration_only_top_p_physical_match"]
    assert isinstance(top_p, dict)
    records = top_p["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0][field] = value
    _assert_resealed_summary_rejected(payload, match="hot-byte arithmetic replay")


def test_signed_summary_rejects_online_moment_arithmetic_inconsistency() -> None:
    payload = _minimal_unsigned_summary()
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(system, dict)
    per_arm = system["per_arm"]
    assert isinstance(per_arm, list) and isinstance(per_arm[0], dict)
    moments = per_arm[0]["intent_to_treat_accuracy"]
    assert isinstance(moments, dict)
    moments["mean"] = 0.5
    _assert_resealed_summary_rejected(payload, match="arithmetic replay")


def test_signed_summary_rejects_regime_subtotal_inconsistency() -> None:
    payload = _minimal_unsigned_summary()
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(system, dict)
    per_arm = system["per_arm"]
    assert isinstance(per_arm, list) and isinstance(per_arm[0], dict)
    regimes = per_arm[0]["signal_diagnostics_by_regime"]
    assert isinstance(regimes, list) and isinstance(regimes[0], dict)
    regimes[0]["clip_saturated_rows"] = 1
    _assert_resealed_summary_rejected(payload, match="regime subtotal replay")


def test_signed_summary_rejects_cross_slice_mean_inconsistency() -> None:
    payload = _minimal_unsigned_summary()
    quality = payload["paired_quality_statistics"]
    assert isinstance(quality, dict)
    primary = quality[summary.PRIMARY_QUALITY_METRIC]
    assert isinstance(primary, dict)
    central = primary["original_central_causal"]
    assert isinstance(central, dict)
    contrast = central[contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME]
    assert isinstance(contrast, dict)
    contexts = contrast["by_context"]
    assert isinstance(contexts, list) and isinstance(contexts[0], dict)
    contexts[0].update(
        {
            "candidate_mean": 0.61,
            "mean_difference": 0.11,
            "mean_difference_percentage_points": 11.0,
            "absolute_mean_difference_replayed": 0.11,
            "relative_mean_difference": 0.22,
            "relative_mean_difference_percent": 22.0,
        }
    )
    _assert_resealed_summary_rejected(payload, match="aggregation drifted")


def test_flat_matrix_record_reconstructs_evaluator_coordinate() -> None:
    training_seed = contract.TRAINING_SEEDS[0]
    _, calibration_seed, evaluation_seed = contract.seed_triplet(training_seed)
    record = {
        "scale": "s55",
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "evaluation_seed": evaluation_seed,
        "budget": "2x",
        "family": contract.FAMILIES[0],
        "context": contract.CONTEXTS[0],
        "replicate": contract.REPLICATES[0],
        "generation_seed": contract.generation_seed(
            evaluation_seed,
            contract.FAMILIES[0],
            contract.CONTEXTS[0],
            contract.REPLICATES[0],
        ),
    }
    coordinate = summary._coordinate_from_record(record)
    assert coordinate["global_block_budget"] == contract.DIRECT_GLOBAL_BLOCK_BUDGETS["s55"]["2x"]
    assert coordinate["csa_layers"] == list(contract.DIRECT_CSA_LAYERS_BY_SCALE["s55"])


def test_frozen_analysis_provenance_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        summary.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    digest = "a" * 64
    current = {"commit": "c" * 40, "dirty": False}
    integrity = {"commit": "d" * 40, "dirty": False}
    manifest = {
        "implementation_digest": digest,
        "implementation_source_commit": "b" * 40,
        "attestation": attestation.public_manifest_contract(TRUST_ROOT.key_id),
    }
    validated = summary.validate_frozen_analysis_provenance(
        implementation_tree_sha256=digest,
        current_source_state=current,
        integrity_source_state=integrity,
        manifest_binding=manifest,
        trust_root=TRUST_ROOT,
    )
    assert validated["implementation_digest_matches_manifest"] is True

    with pytest.raises(ValueError, match="implementation differs"):
        summary.validate_frozen_analysis_provenance(
            implementation_tree_sha256="e" * 64,
            current_source_state=current,
            integrity_source_state=integrity,
            manifest_binding=manifest,
            trust_root=TRUST_ROOT,
        )
    with pytest.raises(ValueError, match="clean"):
        summary.validate_frozen_analysis_provenance(
            implementation_tree_sha256=digest,
            current_source_state={**current, "dirty": True},
            integrity_source_state=integrity,
            manifest_binding=manifest,
            trust_root=TRUST_ROOT,
        )
    with pytest.raises(ValueError, match="trust root"):
        summary.validate_frozen_analysis_provenance(
            implementation_tree_sha256=digest,
            current_source_state=current,
            integrity_source_state=integrity,
            manifest_binding=manifest,
            trust_root=WRONG_TRUST_ROOT,
        )
    monkeypatch.setattr(
        summary.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(ValueError, match="does not descend"):
        summary.validate_frozen_analysis_provenance(
            implementation_tree_sha256=digest,
            current_source_state=current,
            integrity_source_state=integrity,
            manifest_binding=manifest,
            trust_root=TRUST_ROOT,
        )


def _replay_minimal_decision(payload: dict[str, object]) -> dict[str, object]:
    quality = payload["paired_quality_statistics"]
    assert isinstance(quality, dict)
    primary = quality[summary.PRIMARY_QUALITY_METRIC]
    assert isinstance(primary, dict)
    central = primary["original_central_causal"]
    direct = primary["confirmatory"]
    assert isinstance(central, dict) and isinstance(direct, dict)
    system = payload["physical_and_reference_system_statistics"]
    assert isinstance(system, dict)
    return summary.decision_summary({**central, **direct}, system_summary=system)


@pytest.mark.parametrize(
    ("mutation", "field"),
    (
        ("mean", "mean_difference"),
        ("seeds", "all_five_seed_effects_positive"),
        ("conversation_ci", "paired_bootstrap_ci"),
        ("seed_ci", "seed_cluster_bootstrap_ci"),
        ("simultaneous_ci", "bonferroni_four_cell_seed_cluster_bootstrap_ci"),
        ("failure", "candidate_technical_failures"),
        ("physical", "exact_physical_hot_byte_parity"),
    ),
)
def test_every_central_gate_clause_is_required(mutation: str, field: str) -> None:
    payload = _minimal_unsigned_summary()
    assert _replay_minimal_decision(payload)["overall_verdict"] == "GO-OBSERVED-COHORT-BOUNDED"
    quality = payload["paired_quality_statistics"]
    assert isinstance(quality, dict)
    primary = quality[summary.PRIMARY_QUALITY_METRIC]
    assert isinstance(primary, dict)
    central = primary["original_central_causal"]
    assert isinstance(central, dict)
    contrast = central[contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME]
    assert isinstance(contrast, dict)
    cells = contrast["cells"]
    assert isinstance(cells, list)
    cell = cells[0]
    assert isinstance(cell, dict)
    if mutation == "mean":
        cell[field] = -0.01
    elif mutation == "seeds":
        cell[field] = False
    elif mutation == "conversation_ci":
        cell[field] = [-0.01, 0.2]
    elif mutation in {"seed_ci", "simultaneous_ci"}:
        inference = cell["seed_cluster_inference"]
        assert isinstance(inference, dict)
        inference[field] = [-0.01, 0.2]
    elif mutation == "failure":
        cell[field] = 1
    else:
        system = payload["physical_and_reference_system_statistics"]
        assert isinstance(system, dict)
        rows = system["confirmatory_paired_hot_byte_parity"]
        assert isinstance(rows, list)
        target = next(
            row
            for row in rows
            if row["contrast"] == contract.ORIGINAL_CENTRAL_CAUSAL_CONTRAST_NAME
            and row["scale"] == cell["scale"]
            and row["budget"] == cell["budget"]
        )
        target[field] = False
    decision = _replay_minimal_decision(payload)
    assert decision["original_central_causal_gate_pass"] is False
    assert decision["both_hsoft_comparators_pass_observed_cohort_gate"] is True
    assert decision["overall_verdict"] == "NO-GO-CONFIRMATORY"


def test_hsoft_failure_blocks_overall_but_top_p_cannot_change_it() -> None:
    payload = _minimal_unsigned_summary()
    quality = payload["paired_quality_statistics"]
    assert isinstance(quality, dict)
    primary = quality[summary.PRIMARY_QUALITY_METRIC]
    assert isinstance(primary, dict)
    direct = primary["confirmatory"]
    assert isinstance(direct, dict)
    first = direct[summary.CONFIRMATORY_CONTRASTS[0].name]
    assert isinstance(first, dict)
    cells = first["cells"]
    assert isinstance(cells, list) and isinstance(cells[0], dict)
    cells[0]["all_five_seed_effects_positive"] = False
    top_p = primary["top_p_descriptive_sensitivity"]
    assert isinstance(top_p, dict)
    top_p.clear()
    decision = _replay_minimal_decision(payload)
    assert decision["original_central_causal_gate_pass"] is True
    assert decision["both_hsoft_comparators_pass_observed_cohort_gate"] is False
    assert decision["overall_verdict"] == "NO-GO-CONFIRMATORY"
