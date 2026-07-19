from __future__ import annotations

import copy
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_direct_controller_contract as contract  # noqa: E402
from freeze_p2_causal_factorial_arms import (  # noqa: E402
    SUPPLEMENTAL_BASELINE_ARMS,
    build_arm_configs,
)

TEST_TRUST_ROOT = contract.attestation.TrustRoot(
    key=bytes(range(32)),
    key_id=contract.attestation.derive_key_id(bytes(range(32))),
)
ALTERNATE_TRUST_ROOT = contract.attestation.TrustRoot(
    key=bytes(range(32, 64)),
    key_id=contract.attestation.derive_key_id(bytes(range(32, 64))),
)


def _attest_payload(
    payload: dict[str, Any],
    *,
    trust_root: contract.attestation.TrustRoot,
    purpose: str,
) -> dict[str, Any]:
    payload.pop("attestation", None)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = contract.json_digest(payload)
    payload["attestation"] = contract.attestation.attest_payload(
        payload,
        trust_root=trust_root,
        purpose=purpose,
    )
    return payload


def _calibration(*, trust_root: contract.attestation.TrustRoot = TEST_TRUST_ROOT) -> dict[str, Any]:
    def cell(budgets: list[list[int]], digest: str) -> dict[str, Any]:
        total = sum(value for _, value in budgets)
        return {
            "requested_global_budget": total,
            "csa_layer_count": len(budgets),
            "signal_config": {
                "global_block_budget": total,
                "dense_fallback_block_budget": total,
            },
            "quota": {"layer_budgets": budgets, "calibration_digest": digest},
            "identifiability": {
                "terminal_decision": "GO",
                "all_requested_budgets_exact": True,
                "nonbaseline_gate_passed": True,
                "distinct_quota_gate_passed": True,
                "successful_plan_count": 100,
                "requested_plan_count": 100,
                "failures": [],
            },
            "terminal_decision": "GO",
        }

    payload = {
        "experiment_id": contract.DIRECT_CALIBRATION_EXPERIMENT_ID,
        "scale": "s55",
        "training_seed": contract.TRAINING_SEEDS[0],
        "seed": contract.CALIBRATION_SEEDS[0],
        "calibration_seed": contract.CALIBRATION_SEEDS[0],
        "evaluation_seed_reserved": contract.EVALUATION_SEEDS[0],
        "status": "terminal",
        "terminal_decision": "GO",
        "budget_decisions": {"2x": "GO", "4x": "GO"},
        "source": {"commit": "a" * 40, "dirty": False},
        "manifest": {
            "path": "/tmp/direct-manifest.json",
            "sha256": "b" * 64,
            "experiment_id": contract.EXPERIMENT_ID,
            "implementation_digest": "c" * 64,
            "implementation_source_commit": "d" * 40,
            "attestation": contract.attestation.public_manifest_contract(trust_root.key_id),
        },
        "checkpoint": {
            "path": "/tmp/direct-checkpoint.pt",
            "sha256": "e" * 64,
            "bytes": 1024,
        },
        "calibrations": {
            "2x": cell([[2, 2], [4, 4], [6, 6]], "1" * 64),
            "4x": cell([[2, 4], [4, 8], [6, 12]], "2" * 64),
        },
    }
    return _attest_payload(
        payload,
        trust_root=trust_root,
        purpose="p2-direct-soft-lag-calibration-v1",
    )


def _resign_calibration(
    payload: dict[str, Any],
    *,
    trust_root: contract.attestation.TrustRoot = TEST_TRUST_ROOT,
) -> dict[str, Any]:
    return _attest_payload(
        payload,
        trust_root=trust_root,
        purpose="p2-direct-soft-lag-calibration-v1",
    )


def _match_artifact(
    comparator: str,
    *,
    calibration: Mapping[str, Any],
    budget: str,
    low: int,
    high: int,
    numerator: int,
    denominator: int = 4,
    trust_root: contract.attestation.TrustRoot = TEST_TRUST_ROOT,
) -> dict[str, Any]:
    cell = calibration["calibrations"][budget]
    global_budget = cell["requested_global_budget"]
    csa_layers = [layer for layer, _ in cell["quota"]["layer_budgets"]]
    observations = []
    for index in range(denominator):
        high_selected = (index + 1) * numerator // denominator > index * numerator // denominator
        observations.append(
            {
                "observation_index": index,
                "schedule_variant": "high" if high_selected else "low",
                "target_hot_resident_bytes": 100,
                "comparator_hot_resident_bytes": 100,
                "target_is_cuda_hbm_evidence": True,
                "comparator_is_cuda_hbm_evidence": True,
            }
        )
    payload = {
        "schema_version": 1,
        "experiment_id": contract.DIRECT_TOP_P_MATCH_EXPERIMENT_ID,
        "artifact_type": "calibration-only-physical-hot-byte-match",
        "status": "terminal",
        "terminal_decision": "GO",
        "comparator_arm": comparator,
        "target_arm": contract.PRIMARY_ADAPTIVE_ARM,
        "target_metric": contract.PHYSICAL_MATCH_TARGET_METRIC,
        "execution_path": contract.EXECUTION_PATH,
        "mixture_rule": contract.FIXED_MIXTURE_RULE,
        "match_scope": "calibration-only-mean-hot-bytes",
        "fill_contract": "variable-top-p-no-exact-fill",
        "eligible_for_primary_comparison": False,
        "scale": calibration["scale"],
        "training_seed": calibration["training_seed"],
        "calibration_seed": calibration["calibration_seed"],
        "budget": budget,
        "global_block_budget": global_budget,
        "csa_layers": csa_layers,
        "top_p": contract.EXPECTED_ARM_SEMANTICS[comparator].top_p,
        "calibration_payload_sha256": calibration["payload_sha256"],
        "calibration_digest": cell["quota"]["calibration_digest"],
        "source": calibration["source"],
        "manifest": calibration["manifest"],
        "checkpoint": calibration["checkpoint"],
        "schedule": {
            "uniform_low_blocks_per_layer": low,
            "uniform_high_blocks_per_layer": high,
            "mixture_high_numerator": numerator,
            "mixture_denominator": denominator,
        },
        "raw_physical_observations": observations,
        "summary": {
            "observation_count": len(observations),
            "target_hot_resident_bytes_total": 100 * len(observations),
            "comparator_hot_resident_bytes_total": 100 * len(observations),
            "relative_difference": 0.0,
        },
        "audit": {
            "payload_digest_verified": True,
            "external_bindings_verified": True,
            "raw_observations_replayed": True,
            "cuda_hbm_bytes_verified": True,
            "deterministic_schedule_verified": True,
            "calibration_only_scope_verified": True,
        },
    }
    return _attest_payload(
        payload,
        trust_root=trust_root,
        purpose=contract.TOP_P_MATCH_ATTESTATION_PURPOSE,
    )


def _comparator_matches(
    calibration: Mapping[str, Any] | None = None,
    *,
    budget: str = "2x",
    trust_root: contract.attestation.TrustRoot = TEST_TRUST_ROOT,
) -> dict[str, dict[str, Any]]:
    calibration = _calibration() if calibration is None else calibration
    return {
        "fixed-top-p-0.5+pins": _match_artifact(
            "fixed-top-p-0.5+pins",
            calibration=calibration,
            budget=budget,
            low=2,
            high=3,
            numerator=2,
            trust_root=trust_root,
        ),
        "fixed-top-p-0.8+pins": _match_artifact(
            "fixed-top-p-0.8+pins",
            calibration=calibration,
            budget=budget,
            low=3,
            high=4,
            numerator=3,
            trust_root=trust_root,
        ),
    }


def _resign_match(
    payload: dict[str, Any],
    *,
    trust_root: contract.attestation.TrustRoot = TEST_TRUST_ROOT,
) -> dict[str, Any]:
    return _attest_payload(
        payload,
        trust_root=trust_root,
        purpose=contract.TOP_P_MATCH_ATTESTATION_PURPOSE,
    )


def _quality_build(
    calibration: Mapping[str, Any],
    budget: str = "2x",
    *,
    comparator_matches: Mapping[str, Mapping[str, Any]] | None = None,
    strict_matching: bool = True,
    expected_scale: str = "s55",
    expected_training_seed: int = contract.TRAINING_SEEDS[0],
    expected_global_block_budget: int | None = None,
    expected_csa_layers: tuple[int, ...] | None = None,
    trust_root: contract.attestation.TrustRoot = TEST_TRUST_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = (
        _comparator_matches(calibration, budget=budget, trust_root=trust_root)
        if comparator_matches is None
        else comparator_matches
    )
    global_budget = (
        contract.DIRECT_GLOBAL_BLOCK_BUDGETS[expected_scale][budget]
        if expected_global_block_budget is None
        else expected_global_block_budget
    )
    csa_layers = (
        contract.DIRECT_CSA_LAYERS_BY_SCALE[expected_scale]
        if expected_csa_layers is None
        else expected_csa_layers
    )
    return contract.build_direct_controller_arms(
        calibration,
        budget,
        trust_root=trust_root,
        expected_scale=expected_scale,
        expected_training_seed=expected_training_seed,
        expected_global_block_budget=global_budget,
        expected_csa_layers=csa_layers,
        comparator_matches=matches,
        strict_matching=strict_matching,
    )


@pytest.fixture
def stub_full_calibration_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[Mapping[str, Any], bool, contract.attestation.TrustRoot]]:
    """Keep unit artifacts small while spying on the mandatory full-validator API."""

    calls: list[tuple[Mapping[str, Any], bool, contract.attestation.TrustRoot]] = []

    def validator(
        payload: Mapping[str, Any],
        *,
        verify_bindings: bool = False,
        trust_root: contract.attestation.TrustRoot,
    ) -> dict[str, Any]:
        calls.append((payload, verify_bindings, trust_root))
        manifest = payload.get("manifest")
        assert isinstance(manifest, Mapping)
        manifest_attestation = manifest.get("attestation")
        assert isinstance(manifest_attestation, Mapping)
        if manifest_attestation.get("key_id") != trust_root.key_id:
            raise ValueError("Calibration attestation key ID drifted.")
        envelope = payload.get("attestation")
        assert isinstance(envelope, Mapping)
        semantic = dict(payload)
        semantic.pop("attestation")
        contract.attestation.verify_attestation(
            semantic,
            envelope,
            trust_root=trust_root,
            purpose="p2-direct-soft-lag-calibration-v1",
        )
        return dict(payload)

    def loader() -> Callable[..., dict[str, Any]]:
        return validator

    def physical_validator(
        payload: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        trust_root = kwargs.get("trust_root")
        assert isinstance(trust_root, contract.attestation.TrustRoot)
        envelope = payload.get("attestation")
        assert isinstance(envelope, Mapping)
        semantic = dict(payload)
        semantic.pop("attestation")
        contract.attestation.verify_attestation(
            semantic,
            envelope,
            trust_root=trust_root,
            purpose=contract.TOP_P_MATCH_ATTESTATION_PURPOSE,
        )
        return dict(payload)

    def physical_loader() -> Callable[..., dict[str, Any]]:
        return physical_validator

    monkeypatch.setattr(contract, "_direct_calibration_validator", loader)
    monkeypatch.setattr(contract, "_direct_physical_match_validator", physical_loader)
    return calls


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": contract.EXPERIMENT_ID,
        "status": contract.MANIFEST_STATUS,
        "attestation": contract.attestation.public_manifest_contract("e" * 64),
        "adaptation_disclosure": {
            "post_707_rank_no_go": True,
            "prior_p2_quality_results_observed": True,
        },
        "cohort": {
            "training_seeds": list(contract.TRAINING_SEEDS),
            "calibration_seeds": list(contract.CALIBRATION_SEEDS),
            "evaluation_seeds": list(contract.EVALUATION_SEEDS),
            "seed_namespaces_pairwise_disjoint": True,
            "fresh_evaluation_namespace": True,
        },
        "grid": {
            "scales": list(contract.SCALES),
            "budgets": list(contract.BUDGETS),
            "families": list(contract.FAMILIES),
            "contexts": list(contract.CONTEXTS),
            "replicates": list(contract.REPLICATES),
            "examples_per_shard": contract.EXAMPLES_PER_SHARD,
            "generation_seed_rule": contract.GENERATION_SEED_RULE,
            "paired_across_scales_budgets_and_arms": True,
            "cardinalities": contract.expected_grid_cardinalities(),
        },
        "phases": {
            "phase_a_confirmatory_set": list(contract.CONFIRMATORY_ARM_NAMES),
            "phase_a_pareto_sensitivities": list(contract.SENSITIVITY_COMPARATOR_ARMS),
            "phase_a_all": list(contract.PHASE_A_ARM_NAMES),
            "phase_b_diagnostic": list(contract.PHASE_B_DIAGNOSTIC_ARM_NAMES),
            "all_arms": list(contract.ALL_ARM_NAMES),
            "arm_features": contract.expected_arm_features(),
        },
        "primary_estimand": {
            "adaptive_arm": contract.PRIMARY_ADAPTIVE_ARM,
            "confirmatory_comparators": list(contract.CONFIRMATORY_COMPARATOR_ARMS),
            "confirmatory_estimands": contract.CONFIRMATORY_ESTIMANDS,
            "confirmatory_decision_rule": contract.CONFIRMATORY_DECISION_RULE,
            "pareto_sensitivity_comparators": list(contract.SENSITIVITY_COMPARATOR_ARMS),
            "comparator_selection_from_outcomes": False,
            "strongest_fixed_comparator_rule": contract.STRONGEST_FIXED_COMPARATOR_RULE,
            "top_p_sensitivity_rule": contract.TOP_P_SENSITIVITY_RULE,
            "top_p_calibration_only_mean_hot_byte_matching": True,
            "top_p_eligible_for_primary_comparison": False,
            "primary_exact_fill_required": True,
            "memory_match_target_metric": contract.PHYSICAL_MATCH_TARGET_METRIC,
        },
        "execution_contract": {
            "literal_model_path": contract.EXECUTION_PATH,
            "same_literal_path_for_every_arm": True,
            "batch_size": contract.BATCH_SIZE,
            "decode_tokens_per_step": contract.DECODE_TOKENS_PER_STEP,
            "single_token_decode": True,
            "exact_fill_arms": list(contract.EXACT_FILL_ARM_NAMES),
            "variable_fill_sensitivity_arms": list(contract.VARIABLE_FILL_SENSITIVITY_ARMS),
            "exact_fill_rule": contract.EXACT_FILL_RULE,
            "per_layer_hot_floor": contract.PER_LAYER_HOT_FLOOR,
            "zero_cap_tier_stores_forbidden": True,
            "physical_audit_rule": contract.PHYSICAL_AUDIT_RULE,
            "soft_lag_signal_rule": contract.SOFT_LAG_SIGNAL_RULE,
            "signal_diagnostic_rule": contract.SIGNAL_DIAGNOSTIC_RULE,
            "signal_weight_rule": contract.expected_signal_weight_rule(),
            "legacy_configs_are_scaffolds_only": True,
            "direct_arm_semantics_authoritative": True,
            "balanced_feasible_control_rule": contract.BALANCED_FEASIBLE_CONTROL_RULE,
            "configured_caps_are_physical_evidence": False,
            "actual_hot_tensor_bytes_recorded_per_token": True,
            "cuda_peak_allocated_and_reserved_recorded": True,
            "pin_ids_and_counts_bound_before_resize": True,
            "fallback_boundary": contract.FALLBACK_BOUNDARY,
            "fallback_preserves_exact_b": True,
            "chunked_equivalence_required": False,
            "outcome_dependent_early_stopping": False,
            "phase_b_runs_regardless_of_phase_a_outcomes": True,
        },
        "implementation": {
            "paths": list(contract.IMPLEMENTATION_PATHS),
            "tree_digest": "a" * 64,
            "source_commit": "b" * 40,
        },
    }


def _schedule_totals(arm: Any) -> list[int]:
    return [sum(value for _, value in config.layer_budgets) for config in arm.configs]


def test_seed_grid_and_conversation_cardinalities_are_paper_grade() -> None:
    contract.validate_seed_namespaces()

    assert contract.TRAINING_SEEDS == tuple(range(6071406, 6071411))
    assert contract.CALIBRATION_SEEDS == tuple(range(7071406, 7071411))
    assert contract.EVALUATION_SEEDS == tuple(range(10071406, 10071411))
    assert contract.seed_triplet(6071410) == (6071410, 7071410, 10071410)
    assert len(contract.FAMILIES) == 9
    assert contract.EXAMPLES_PER_FAMILY == 1_000
    assert contract.UNIQUE_SHARDS_TOTAL == 4_500
    assert contract.BUDGET_SHARDS_TOTAL == 9_000
    assert contract.expected_grid_cardinalities()["all_arm_conversations"] == 3_420_000


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_direct_builder_requires_new_calibration_id_and_exact_global_b() -> None:
    legacy = _calibration()
    legacy["experiment_id"] = contract.LEGACY_CALIBRATION_SCAFFOLD_ID
    _resign_calibration(legacy)
    with pytest.raises(ValueError, match="direct soft-lag calibration"):
        _quality_build(legacy)

    mismatched = _calibration()
    mismatched["calibrations"]["2x"]["signal_config"]["dense_fallback_block_budget"] += 1
    _resign_calibration(mismatched)
    with pytest.raises(ValueError, match="exactly equal"):
        _quality_build(mismatched)


@pytest.mark.parametrize("gate", ("top-level", "budget-cell"))
@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_direct_builder_rejects_digest_bound_target_free_no_go(gate: str) -> None:
    calibration = _calibration()
    if gate == "top-level":
        calibration["terminal_decision"] = "NO-GO"
    else:
        calibration["calibrations"]["2x"]["terminal_decision"] = "NO-GO"
        calibration["calibrations"]["2x"]["identifiability"]["terminal_decision"] = "NO-GO"
    _resign_calibration(calibration)

    with pytest.raises(ValueError, match="GO gate|identifiability gate"):
        _quality_build(calibration)


def test_direct_builder_rejects_self_hashed_envelope_without_full_artifact() -> None:
    """A digest-valid local envelope cannot stand in for deterministic full replay."""

    with pytest.raises(ValueError, match="Calibration schema drifted"):
        _quality_build(_calibration())


def test_direct_builder_mandates_full_validator_with_live_bindings(
    stub_full_calibration_validator: list[
        tuple[Mapping[str, Any], bool, contract.attestation.TrustRoot]
    ],
) -> None:
    calibration = _calibration()

    _quality_build(calibration)

    assert stub_full_calibration_validator == [(calibration, True, TEST_TRUST_ROOT)]


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_direct_builder_accepts_genuine_hmac_attested_calibration() -> None:
    calibration = _calibration()
    envelope = calibration["attestation"]
    semantic = dict(calibration)
    semantic.pop("attestation")

    contract.attestation.verify_attestation(
        semantic,
        envelope,
        trust_root=TEST_TRUST_ROOT,
        purpose="p2-direct-soft-lag-calibration-v1",
    )
    arms, _ = _quality_build(calibration)

    assert contract.PRIMARY_ADAPTIVE_ARM in arms


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_direct_builder_fail_closes_expected_launch_coordinate() -> None:
    calibration = _calibration()

    with pytest.raises(ValueError, match="scale/launch mismatch"):
        _quality_build(calibration, expected_scale="s151")
    with pytest.raises(ValueError, match="training-seed/launch mismatch"):
        _quality_build(calibration, expected_training_seed=contract.TRAINING_SEEDS[1])
    with pytest.raises(ValueError, match="global B drifted from the frozen scale/budget"):
        _quality_build(calibration, expected_global_block_budget=11)
    with pytest.raises(ValueError, match="CSA layer inventory drifted from the frozen scale"):
        _quality_build(calibration, expected_csa_layers=(2, 4, 8))

    drifted_layers = _calibration()
    drifted_layers["calibrations"]["2x"]["quota"]["layer_budgets"] = [
        [2, 2],
        [4, 4],
        [8, 6],
    ]
    _resign_calibration(drifted_layers)
    with pytest.raises(ValueError, match="CSA layer inventory/launch mismatch"):
        _quality_build(drifted_layers)

    drifted_requested_budget = _calibration()
    drifted_requested_budget["calibrations"]["2x"]["requested_global_budget"] = 11
    _resign_calibration(drifted_requested_budget)
    with pytest.raises(ValueError, match="requested global B/launch mismatch"):
        _quality_build(drifted_requested_budget)


def test_generation_seed_is_fresh_deterministic_unique_and_arm_paired() -> None:
    seeds = {
        contract.generation_seed(evaluation_seed, family, context, replicate)
        for evaluation_seed in contract.EVALUATION_SEEDS
        for family in contract.FAMILIES
        for context in contract.CONTEXTS
        for replicate in contract.REPLICATES
    }

    assert len(seeds) == 5 * 9 * 5 * 10
    first = contract.generation_seed(10071406, contract.FAMILIES[0], 80, 0)
    assert first == contract.generation_seed(10071406, contract.FAMILIES[0], 80, 0)
    assert seeds.isdisjoint(contract.TRAINING_SEEDS)
    assert seeds.isdisjoint(contract.CALIBRATION_SEEDS)
    assert seeds.isdisjoint(contract.EVALUATION_SEEDS)
    with pytest.raises(ValueError, match="Unregistered direct-controller evaluation seed"):
        contract.generation_seed(8071401, contract.FAMILIES[0], 80, 0)


def test_primary_and_diagnostic_phases_have_exactly_nineteen_unique_arms() -> None:
    assert contract.CONFIRMATORY_ARM_NAMES == (
        "hierarchical-soft-lag+pins",
        "fixed+pins",
        "hierarchical-balanced-fixed+pins",
    )
    assert contract.PHASE_A_ARM_NAMES == (
        "hierarchical-soft-lag+pins",
        "fixed+pins",
        "hierarchical-balanced-fixed+pins",
        "fixed-top-p-0.5+pins",
        "fixed-top-p-0.8+pins",
    )
    assert len(contract.PHASE_B_DIAGNOSTIC_ARM_NAMES) == 14
    assert set(contract.PHASE_A_ARM_NAMES).isdisjoint(contract.PHASE_B_DIAGNOSTIC_ARM_NAMES)
    assert len(contract.ALL_ARM_NAMES) == len(set(contract.ALL_ARM_NAMES)) == 19
    assert set(contract.VARIABLE_FILL_SENSITIVITY_ARMS).isdisjoint(contract.EXACT_FILL_ARM_NAMES)
    assert len(contract.EXACT_FILL_ARM_NAMES) == 17


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_builder_pins_top_p_without_mutating_legacy_arm_globals() -> None:
    legacy_names_before = tuple(arm.name for arm in SUPPLEMENTAL_BASELINE_ARMS)
    arms, metadata = _quality_build(_calibration())
    legacy_calibration = _calibration()
    legacy_calibration["experiment_id"] = contract.LEGACY_CALIBRATION_SCAFFOLD_ID
    legacy, _ = build_arm_configs(legacy_calibration, "2x")

    assert tuple(arms) == contract.ALL_ARM_NAMES
    assert tuple(arm.name for arm in SUPPLEMENTAL_BASELINE_ARMS) == legacy_names_before
    assert arms["fixed-top-p-0.5+pins"].spec.protected_pins is True
    assert arms["fixed-top-p-0.8+pins"].spec.protected_pins is True
    assert all(
        config.enable_protected_pins
        for name in ("fixed-top-p-0.5+pins", "fixed-top-p-0.8+pins")
        for config in arms[name].configs
    )
    assert all(
        not config.enable_protected_pins
        for name in ("fixed-top-p-0.5", "fixed-top-p-0.8")
        for config in legacy[name].configs
    )
    assert metadata["strict_sensitivity_matching"] is True
    assert metadata["quality_launch_validated"] is True
    assert arms[contract.PRIMARY_ADAPTIVE_ARM].spec.name == contract.PRIMARY_ADAPTIVE_ARM
    assert (
        contract.EXPECTED_ARM_SEMANTICS[contract.PRIMARY_ADAPTIVE_ARM].quota_runtime == "soft-lag"
    )
    assert contract.EXPECTED_ARM_SEMANTICS[contract.PRIMARY_ADAPTIVE_ARM].lag_tokens == 1
    contract.validate_arm_semantics(arms)


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_confirmatory_controls_are_exact_b_while_top_p_uses_mean_matches() -> None:
    arms, metadata = _quality_build(_calibration())

    assert _schedule_totals(arms["fixed+pins"]) == [12]
    assert _schedule_totals(arms[contract.CLEAN_ALLOCATOR_CONTROL_ARM]) == [12]
    assert _schedule_totals(arms["fixed"]) == [12]
    assert _schedule_totals(arms["fixed-top-p-0.5+pins"]) == [6, 9]
    assert _schedule_totals(arms["fixed-top-p-0.8+pins"]) == [9, 12]
    assert arms["fixed"].mixture_high_numerator == 0
    assert arms["fixed+pins"].mixture_high_numerator == 0
    assert arms[contract.CLEAN_ALLOCATOR_CONTROL_ARM].mixture_high_numerator == 0
    assert metadata["physical_match_target_arm"] == "hierarchical-soft-lag+pins"
    assert metadata["physical_match_target_metric"] == "hot_resident_bytes"
    assert metadata["signal_diagnostic_rule"] == contract.SIGNAL_DIAGNOSTIC_RULE
    schedules = metadata["top_p_sensitivity_schedules"]
    assert schedules["fixed-top-p-0.5+pins"]["mixture_high_numerator"] == 2
    assert schedules["fixed-top-p-0.8+pins"]["mixture_high_numerator"] == 3
    assert metadata["strongest_fixed_comparator_rule"] == (contract.STRONGEST_FIXED_COMPARATOR_RULE)
    assert tuple(metadata["confirmatory_comparators"]) == (
        "fixed+pins",
        "hierarchical-balanced-fixed+pins",
    )
    assert metadata["confirmatory_estimands"] == contract.CONFIRMATORY_ESTIMANDS
    assert metadata["confirmatory_decision_rule"] == contract.CONFIRMATORY_DECISION_RULE


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_strict_matching_rejects_missing_comparator_schedule() -> None:
    matches = _comparator_matches()
    matches.pop("fixed-top-p-0.8+pins")

    with pytest.raises(ValueError, match="inventory is incomplete"):
        _quality_build(_calibration(), comparator_matches=matches)
    with pytest.raises(ValueError, match="requires all top-p sensitivity schedules"):
        contract.build_direct_controller_arms(
            _calibration(),
            "2x",
            trust_root=TEST_TRUST_ROOT,
            expected_scale="s55",
            expected_training_seed=contract.TRAINING_SEEDS[0],
            expected_global_block_budget=12,
            expected_csa_layers=(2, 4, 6),
            comparator_matches=None,
        )
    with pytest.raises(ValueError, match="requires strict top-p sensitivity matching"):
        _quality_build(_calibration(), strict_matching=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("target_arm", "calibrated+pins", "target arm drifted"),
        ("target_metric", "logical_blocks", "target metric drifted"),
        ("execution_path", "chunked", "execution path drifted"),
        ("mixture_rule", "random", "mixture rule drifted"),
        ("match_scope", "evaluation", "match scope drifted"),
        ("fill_contract", "exact-fill", "fill contract drifted"),
        ("eligible_for_primary_comparison", True, "became eligible"),
    ),
)
@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_strict_matching_rejects_provenance_drift(field: str, value: str, message: str) -> None:
    matches = _comparator_matches()
    matches["fixed-top-p-0.5+pins"][field] = value
    _resign_match(matches["fixed-top-p-0.5+pins"])

    with pytest.raises(ValueError, match=message):
        _quality_build(_calibration(), comparator_matches=matches)


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_strict_matching_rejects_calibration_digest_and_tolerance_drift() -> None:
    matches = _comparator_matches()
    matches["fixed-top-p-0.5+pins"]["calibration_digest"] = "9" * 64
    _resign_match(matches["fixed-top-p-0.5+pins"])
    with pytest.raises(ValueError, match="calibration digest drifted"):
        _quality_build(_calibration(), comparator_matches=matches)

    matches = _comparator_matches()
    artifact = matches["fixed-top-p-0.5+pins"]
    for observation in artifact["raw_physical_observations"]:
        observation["comparator_hot_resident_bytes"] = 102
    count = len(artifact["raw_physical_observations"])
    artifact["summary"] = {
        "observation_count": count,
        "target_hot_resident_bytes_total": 100 * count,
        "comparator_hot_resident_bytes_total": 102 * count,
        "relative_difference": 0.02,
    }
    _resign_match(artifact)
    with pytest.raises(ValueError, match="relative difference exceeded"):
        _quality_build(_calibration(), comparator_matches=matches)


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_strict_matching_rejects_extreme_or_nonadjacent_schedule() -> None:
    for low, high in ((1_000, 1_001), (2, 4)):
        matches = _comparator_matches()
        artifact = matches["fixed-top-p-0.5+pins"]
        artifact["schedule"]["uniform_low_blocks_per_layer"] = low
        artifact["schedule"]["uniform_high_blocks_per_layer"] = high
        _resign_match(artifact)
        with pytest.raises(ValueError, match="schedule arithmetic"):
            _quality_build(_calibration(), comparator_matches=matches)


def test_quality_build_fails_closed_without_authoritative_physical_match_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = contract.importlib.import_module

    def missing(name: str) -> Any:
        if name == "validate_p2_direct_top_p_physical_match":
            raise ModuleNotFoundError(name=name)
        return original_import(name)

    monkeypatch.setattr(contract.importlib, "import_module", missing)
    monkeypatch.setattr(
        contract,
        "_direct_calibration_validator",
        lambda: lambda payload, verify_bindings=False, trust_root=None: dict(payload),
    )
    with pytest.raises(RuntimeError, match="quality construction remains blocked"):
        _quality_build(_calibration())


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_quality_build_mandates_authoritative_match_replay_with_live_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def validator(payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        envelope = payload.get("attestation")
        assert isinstance(envelope, Mapping)
        semantic = dict(payload)
        semantic.pop("attestation")
        contract.attestation.verify_attestation(
            semantic,
            envelope,
            trust_root=kwargs["trust_root"],
            purpose=contract.TOP_P_MATCH_ATTESTATION_PURPOSE,
        )
        return dict(payload)

    monkeypatch.setattr(contract, "_direct_physical_match_validator", lambda: validator)
    calibration = _calibration()
    _quality_build(calibration)

    assert len(calls) == len(contract.SENSITIVITY_COMPARATOR_ARMS)
    assert {call["comparator"] for call in calls} == set(contract.SENSITIVITY_COMPARATOR_ARMS)
    assert all(
        call["verify_bindings"] is True
        and call["trust_root"] is TEST_TRUST_ROOT
        and call["calibration"] == calibration
        and call["expected_scale"] == "s55"
        and call["expected_training_seed"] == contract.TRAINING_SEEDS[0]
        and call["expected_calibration_seed"] == contract.CALIBRATION_SEEDS[0]
        and call["expected_global_block_budget"] == 12
        and call["expected_csa_layers"] == (2, 4, 6)
        for call in calls
    )


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_quality_build_rejects_wrong_explicit_trust_root() -> None:
    calibration = _calibration()

    with pytest.raises(ValueError, match="Calibration attestation key ID drifted"):
        _quality_build(calibration, trust_root=ALTERNATE_TRUST_ROOT)


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_top_p_attestation_rejects_wrong_key_and_public_self_rehash() -> None:
    calibration = _calibration()
    wrong_key_matches = _comparator_matches(calibration)
    _resign_match(
        wrong_key_matches["fixed-top-p-0.5+pins"],
        trust_root=ALTERNATE_TRUST_ROOT,
    )
    with pytest.raises(ValueError, match="Attestation key ID drifted"):
        _quality_build(calibration, comparator_matches=wrong_key_matches)

    self_rehashed_matches = _comparator_matches(calibration)
    self_rehashed = self_rehashed_matches["fixed-top-p-0.5+pins"]
    self_rehashed["terminal_decision"] = "NO-GO"
    digest_source = dict(self_rehashed)
    digest_source.pop("attestation")
    digest_source.pop("payload_sha256")
    self_rehashed["payload_sha256"] = contract.json_digest(digest_source)
    with pytest.raises(ValueError, match="Attestation payload checksum drifted"):
        _quality_build(calibration, comparator_matches=self_rehashed_matches)


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_top_p_attestation_rejects_valid_cross_comparator_replay() -> None:
    calibration = _calibration()
    matches = _comparator_matches(calibration)
    matches["fixed-top-p-0.8+pins"] = copy.deepcopy(matches["fixed-top-p-0.5+pins"])

    with pytest.raises(ValueError, match="Comparator match identity drifted"):
        _quality_build(calibration, comparator_matches=matches)


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_soft_lag_components_and_permuted_quota_are_distinct_overlays() -> None:
    arms, _metadata = _quality_build(_calibration())
    semantics = contract.EXPECTED_ARM_SEMANTICS
    primary = semantics[contract.PRIMARY_ADAPTIVE_ARM]

    assert primary.quota_runtime == "soft-lag"
    assert primary.lag_tokens == 1
    assert primary.fill_mode == "exact-feasible-B"
    assert "token t-1 ControllerLayerSignal" in contract.SOFT_LAG_SIGNAL_RULE
    assert "evaluation outcomes are forbidden" in contract.SOFT_LAG_SIGNAL_RULE
    assert semantics["hierarchical-soft-lag+pins-no-score"].score_concentration is False
    assert semantics["hierarchical-soft-lag+pins-no-temporal"].temporal_reuse is False
    assert semantics["hierarchical-soft-lag+pins-no-cross-layer"].cross_layer_signal is False
    assert semantics["hierarchical-soft-lag+pins-no-refresh"].refresh_reuse is False
    assert semantics["hierarchical-soft-lag+pins-no-score"].signal_weights == pytest.approx(
        (0.0, 0.0, 5.0 / 9.0, 4.0 / 9.0)
    )
    assert semantics["hierarchical-soft-lag+pins-no-temporal"].signal_weights == pytest.approx(
        (7.0 / 15.0, 4.0 / 15.0, 0.0, 4.0 / 15.0)
    )
    assert semantics["hierarchical-soft-lag+pins-no-cross-layer"].signal_weights == pytest.approx(
        (0.4375, 0.25, 0.3125, 0.0)
    )
    for name in (
        "hierarchical-soft-lag+pins-no-score",
        "hierarchical-soft-lag+pins-no-temporal",
        "hierarchical-soft-lag+pins-no-cross-layer",
    ):
        config = arms[name].configs[0]
        assert (
            config.signal.entropy_weight,
            config.signal.margin_weight,
            config.signal.temporal_weight,
            config.signal.cross_layer_weight,
        ) == semantics[name].signal_weights
        assert sum(semantics[name].signal_weights) == pytest.approx(1.0)
    permuted = semantics["hierarchical-soft-lag+pins-permuted-quota"]
    assert permuted.quota_runtime == "soft-lag-permuted"
    assert permuted.permute_quota is True
    assert _schedule_totals(arms["hierarchical-soft-lag+pins-permuted-quota"]) == [12]
    assert _schedule_totals(arms[contract.PRIMARY_ADAPTIVE_ARM]) == [12]
    assert semantics["local+pins"].quota_runtime == "local-static-quota"
    assert semantics["hierarchical-soft-lag+pins-no-cross-layer"].quota_runtime == "soft-lag"


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_clean_control_holds_everything_except_allocator_weighting_fixed() -> None:
    arms, metadata = _quality_build(_calibration())
    semantics = contract.EXPECTED_ARM_SEMANTICS
    hsoft = semantics[contract.PRIMARY_ADAPTIVE_ARM]
    clean = semantics[contract.CLEAN_ALLOCATOR_CONTROL_ARM]
    conventional = semantics[contract.CONVENTIONAL_FIXED_COMPARATOR_ARM]

    assert clean.quota_runtime == conventional.quota_runtime == "balanced-feasible"
    assert clean.lag_tokens == 0
    assert hsoft.quota_runtime == "soft-lag" and hsoft.lag_tokens == 1
    for field in (
        "fill_mode",
        "score_concentration",
        "temporal_reuse",
        "cross_layer_signal",
        "refresh_reuse",
        "protected_pins",
        "permute_quota",
        "fallback_mode",
        "top_p",
    ):
        assert getattr(clean, field) == getattr(hsoft, field)
    assert (
        arms[contract.CLEAN_ALLOCATOR_CONTROL_ARM].configs
        == arms[contract.PRIMARY_ADAPTIVE_ARM].configs
    )
    assert (
        arms[contract.CONVENTIONAL_FIXED_COMPARATOR_ARM].configs[0].signal
        == arms[contract.PRIMARY_ADAPTIVE_ARM].configs[0].signal
    )
    assert (
        arms[contract.CONVENTIONAL_FIXED_COMPARATOR_ARM].configs[0].layer_budgets
        == arms[contract.PRIMARY_ADAPTIVE_ARM].configs[0].layer_budgets
    )
    assert semantics["calibrated+pins"].quota_runtime == "calibrated-static"
    assert metadata["balanced_feasible_control_rule"] == (contract.BALANCED_FEASIBLE_CONTROL_RULE)
    assert "exact same candidate_caps, pin_floors" in (contract.BALANCED_FEASIBLE_CONTROL_RULE)
    assert (
        "total Hsoft controller-bundle effect"
        in contract.CONFIRMATORY_ESTIMANDS[contract.CONVENTIONAL_FIXED_COMPARATOR_ARM]
    )
    assert (
        "soft-lag signal-weighted allocator effect"
        in contract.CONFIRMATORY_ESTIMANDS[contract.CLEAN_ALLOCATOR_CONTROL_ARM]
    )
    assert "independently against both" in contract.CONFIRMATORY_DECISION_RULE


def test_top_p_remains_variable_and_is_ineligible_for_strongest_fixed_rule() -> None:
    for name in contract.SENSITIVITY_COMPARATOR_ARMS:
        semantics = contract.EXPECTED_ARM_SEMANTICS[name]
        assert semantics.fill_mode == "variable-top-p"
        assert semantics.analysis_role == "pareto-sensitivity"
        assert name not in contract.EXACT_FILL_ARM_NAMES

    assert contract.CONFIRMATORY_COMPARATOR_ARMS == (
        "fixed+pins",
        "hierarchical-balanced-fixed+pins",
    )
    assert set(contract.CONFIRMATORY_COMPARATOR_ARMS).issubset(contract.EXACT_FILL_ARM_NAMES)
    assert "both use the same target-free balanced feasible" in (
        contract.STRONGEST_FIXED_COMPARATOR_RULE
    )
    assert "never converted to top-k" in contract.TOP_P_SENSITIVITY_RULE


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_balanced_primary_fixed_exact_fills_frozen_budget_without_mixture() -> None:
    calibration = _calibration()
    item = calibration["calibrations"]["2x"]
    item["quota"]["layer_budgets"] = [[2, 1], [4, 5], [6, 6]]
    _resign_calibration(calibration)

    first, metadata = _quality_build(calibration)
    second, _ = _quality_build(calibration)
    fixed = first[contract.CONVENTIONAL_FIXED_COMPARATOR_ARM]
    budgets = tuple(value for _, value in fixed.configs[0].layer_budgets)

    assert sum(budgets) == 12
    assert max(budgets) - min(budgets) <= 1
    assert fixed.mixture_high_numerator == 0
    assert len(fixed.configs) == 1
    assert fixed.configs == second[contract.CONVENTIONAL_FIXED_COMPARATOR_ARM].configs
    assert tuple(metadata["exact_fixed_layer_budgets"]) == fixed.configs[0].layer_budgets
    assert (
        first[contract.CLEAN_ALLOCATOR_CONTROL_ARM].configs
        == first[contract.PRIMARY_ADAPTIVE_ARM].configs
    )
    assert first[contract.CLEAN_ALLOCATOR_CONTROL_ARM].configs[0].layer_budgets == (
        fixed.configs[0].layer_budgets
    )


@pytest.mark.usefixtures("stub_full_calibration_validator")
def test_fallback_is_resident_only_and_cannot_change_exact_b() -> None:
    arms, metadata = _quality_build(_calibration())
    fallback_name = "hierarchical-soft-lag+pins+fallback"
    fallback = arms[fallback_name]
    semantics = contract.EXPECTED_ARM_SEMANTICS[fallback_name]

    assert semantics.fallback_mode == contract.FALLBACK_MODE
    assert semantics.fill_mode == "exact-feasible-B"
    assert all(config.layer_budgets == config.dense_layer_budgets for config in fallback.configs)
    assert _schedule_totals(fallback) == _schedule_totals(arms[contract.PRIMARY_ADAPTIVE_ARM])
    assert metadata["fallback_boundary"] == contract.FALLBACK_BOUNDARY
    assert "may not add a block" in contract.FALLBACK_BOUNDARY


def test_manifest_contract_accepts_only_same_path_outcome_independent_design() -> None:
    payload = _manifest()

    assert contract.validate_manifest_payload(payload) is payload
    round_tripped = json.loads(json.dumps(payload))
    assert contract.validate_manifest_payload(round_tripped) is round_tripped

    mutations: tuple[tuple[str, str, object], ...] = (
        ("execution_contract", "literal_model_path", "chunked"),
        ("execution_contract", "same_literal_path_for_every_arm", False),
        ("execution_contract", "batch_size", 4),
        ("execution_contract", "decode_tokens_per_step", 2),
        ("execution_contract", "configured_caps_are_physical_evidence", True),
        ("execution_contract", "actual_hot_tensor_bytes_recorded_per_token", False),
        ("execution_contract", "pin_ids_and_counts_bound_before_resize", False),
        ("execution_contract", "legacy_configs_are_scaffolds_only", False),
        ("execution_contract", "direct_arm_semantics_authoritative", False),
        ("execution_contract", "signal_diagnostic_rule", "aggregate-only"),
        ("execution_contract", "balanced_feasible_control_rule", "static calibrated"),
        ("execution_contract", "signal_weight_rule", {}),
        ("execution_contract", "chunked_equivalence_required", True),
        ("execution_contract", "outcome_dependent_early_stopping", True),
        ("execution_contract", "phase_b_runs_regardless_of_phase_a_outcomes", False),
        ("primary_estimand", "comparator_selection_from_outcomes", True),
        ("primary_estimand", "top_p_eligible_for_primary_comparison", True),
        ("primary_estimand", "confirmatory_comparators", ["fixed+pins"]),
        ("primary_estimand", "confirmatory_decision_rule", "take outcome maximum"),
    )
    for section, field, value in mutations:
        drifted = copy.deepcopy(payload)
        drifted[section][field] = value
        with pytest.raises(ValueError):
            contract.validate_manifest_payload(drifted)


def test_manifest_contract_rejects_arm_seed_and_cardinality_drift() -> None:
    payload = _manifest()

    drifted = copy.deepcopy(payload)
    drifted["phases"]["arm_features"][contract.PRIMARY_ADAPTIVE_ARM]["cross_layer_signal"] = False
    with pytest.raises(ValueError, match="arm-feature semantics"):
        contract.validate_manifest_payload(drifted)

    drifted = copy.deepcopy(payload)
    drifted["cohort"]["evaluation_seeds"][-1] = 8071401
    with pytest.raises(ValueError, match="Fresh evaluation seeds drifted"):
        contract.validate_manifest_payload(drifted)

    drifted = copy.deepcopy(payload)
    drifted["grid"]["cardinalities"]["all_arm_conversations"] -= 1
    with pytest.raises(ValueError, match="Grid cardinalities drifted"):
        contract.validate_manifest_payload(drifted)


def test_implementation_digest_binds_complete_package_and_exact_research_inventory() -> None:
    tracked_tree = contract.subprocess.run(
        ["git", "ls-files", "-s", "--", contract.PACKAGE_IMPLEMENTATION_ROOT],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    entries = tuple(line for line in tracked_tree.splitlines() if line)
    tracked_paths = {entry.split("\t", 1)[1] for entry in entries}

    assert "nano_deepseek_v4/config.py" in tracked_paths
    assert "nano_deepseek_v4/__init__.py" in tracked_paths
    assert "nano_deepseek_v4/py.typed" in tracked_paths
    assert contract.IMPLEMENTATION_PATHS == (
        contract.PACKAGE_IMPLEMENTATION_ROOT,
        *contract.DIRECT_RESEARCH_IMPLEMENTATION_PATHS,
    )
    assert str(contract.MANIFEST_PATH) not in contract.IMPLEMENTATION_PATHS
    assert {
        "research/adaptive_v4_memory/scripts/train_m1_associative_recall.py",
        "research/adaptive_v4_memory/scripts/run_p2_direct_training_matrix.py",
        "research/adaptive_v4_memory/scripts/calibrate_p2_direct_soft_lag.py",
        "research/adaptive_v4_memory/scripts/run_p2_direct_calibration_matrix.py",
        "research/adaptive_v4_memory/scripts/evaluate_p2_direct_controller_shard.py",
        "research/adaptive_v4_memory/scripts/run_p2_direct_controller_matrix.py",
        "research/adaptive_v4_memory/scripts/audit_p2_direct_controller_integrity.py",
        "research/adaptive_v4_memory/scripts/summarize_p2_direct_controller.py",
    }.issubset(contract.IMPLEMENTATION_PATHS)


def test_implementation_digest_changes_when_config_git_object_changes() -> None:
    tracked_tree = contract.subprocess.run(
        ["git", "ls-files", "-s", "--", contract.PACKAGE_IMPLEMENTATION_ROOT],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    entries = tuple(line for line in tracked_tree.splitlines() if line)
    config_entry = next(
        entry for entry in entries if entry.endswith("\tnano_deepseek_v4/config.py")
    )
    metadata, path = config_entry.split("\t", 1)
    mode, object_id, stage = metadata.split()
    changed_object_id = (
        "f" * len(object_id) if object_id != "f" * len(object_id) else "e" * len(object_id)
    )
    changed_entry = f"{mode} {changed_object_id} {stage}\t{path}"
    changed_entries = tuple(changed_entry if entry == config_entry else entry for entry in entries)

    original = contract._implementation_index_digest(
        (contract.PACKAGE_IMPLEMENTATION_ROOT,), entries
    )
    changed = contract._implementation_index_digest(
        (contract.PACKAGE_IMPLEMENTATION_ROOT,), changed_entries
    )

    assert original != changed


def test_implementation_digest_rejects_inventory_order_drift() -> None:
    with pytest.raises(ValueError, match="inventory or ordering drifted"):
        contract.implementation_tree_digest(tuple(reversed(contract.IMPLEMENTATION_PATHS)))


def test_implementation_digest_rejects_untracked_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_paths = (
        "nano_deepseek_v4/__init__.py",
        *contract.DIRECT_RESEARCH_IMPLEMENTATION_PATHS,
    )
    tracked = "\n".join(f"100644 {'a' * 40} 0\t{path}" for path in sorted(tracked_paths))

    class Result:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def fake_run(command: list[str], **_kwargs: Any) -> Result:
        if "--others" in command:
            return Result("nano_deepseek_v4/untracked_runtime.py\n")
        return Result(tracked)

    monkeypatch.setattr(contract.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Untracked files exist"):
        contract.implementation_tree_digest()
