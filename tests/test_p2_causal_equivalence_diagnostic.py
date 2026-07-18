from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import diagnose_p2_causal_equivalence as diagnostic  # noqa: E402


def _action(
    *,
    layer: int,
    batch: int = 0,
    query: int = 8,
    selected: tuple[int, ...] = (3, 7),
) -> dict[str, Any]:
    return {
        "layer_index": layer,
        "batch_index": batch,
        "query_position": query,
        "selected_end_positions": list(selected),
        "pinned_end_positions": [3],
        "budget_limit": 2,
        "fallback_reason": None,
        "refreshed": True,
        "signal": {"score_concentration": 0.75, "entropy": 0.25},
    }


def _success_observation(
    path_name: str,
    *,
    repeat: int = 0,
    predictions: list[list[int]] | None = None,
    logits: torch.Tensor | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> diagnostic.RunObservation:
    predictions = predictions or [[1, 2]]
    logits = logits if logits is not None else torch.tensor([[[2.0, 1.0], [0.5, 3.0]]])
    actions = actions or [_action(layer=4), _action(layer=2)]
    semantic = diagnostic._semantic_actions(actions)
    selected = diagnostic._selected_position_actions(actions)
    pins = diagnostic._pin_actions(actions)
    fallbacks = diagnostic._fallback_actions(actions)
    action_digest = diagnostic._json_digest(semantic)
    selection_digest = diagnostic._json_digest(selected)
    pin_digest = diagnostic._json_digest(pins)
    fallback_digest = diagnostic._json_digest(fallbacks)
    top1 = logits.argmax(dim=-1)
    token_digests = [
        [diagnostic._tensor_digest(logits[batch, token]) for token in range(logits.shape[1])]
        for batch in range(logits.shape[0])
    ]
    payload = {
        "status": "success",
        "path": path_name,
        "repeat_index": repeat,
        "conversation_ids": ["conversation:0"],
        "predictions": predictions,
        "query_input_positions": [8, 9],
        "top1_top2_margins": [[1.0, 2.5]],
        "query_logits": {"sha256": diagnostic._tensor_digest(logits)},
        "post_prefix_trace": {
            "first_input_position": 8,
            "last_input_position": 9,
            "predicted_token_position_offset": 1,
            "top1_token_ids": top1.tolist(),
            "top1_sha256": diagnostic._tensor_digest(top1),
            "logits": {
                "sha256": diagnostic._tensor_digest(logits),
                "sha256_by_conversation_and_token": token_digests,
            },
        },
        "controller": {
            "path_neutral_actions": semantic,
            "path_neutral_action_sha256": action_digest,
            "path_neutral_selected_position_sha256": selection_digest,
            "path_neutral_pin_set_sha256": pin_digest,
            "path_neutral_fallback_action_sha256": fallback_digest,
            "action_sha256_by_conversation": {"conversation:0": action_digest},
            "selected_position_sha256_by_conversation": {"conversation:0": selection_digest},
            "pin_set_sha256_by_conversation": {"conversation:0": pin_digest},
            "fallback_action_sha256_by_conversation": {"conversation:0": fallback_digest},
            "rows_sha256": "a" * 64,
            "rows": [{"budget_violations": 0}],
            "logical_counters": {
                "selected_queries": 2,
                "finalized_control_points": 2,
                "fallback_control_points": 0,
                "peak_selected_blocks": 2,
            },
            "runtime_replay_digest": "b" * 64,
        },
        "accounting": {"logical_bytes": 100},
        "tier": {"hot_bytes": 50},
    }
    return diagnostic.RunObservation(payload=payload, query_logits=logits)


def _error_observation(path_name: str, *, repeat: int = 0) -> diagnostic.RunObservation:
    return diagnostic._error_observation(
        path=diagnostic._path_by_name(path_name),
        repeat_index=repeat,
        execution_ordinal=repeat,
        error=RuntimeError("query-axis selected-block union exceeds hot budget"),
    )


def test_contract_freezes_known_cells_full_2x2_and_three_repeats(tmp_path: Path) -> None:
    cells = {(cell.budget, cell.context, cell.arm) for cell in diagnostic.KNOWN_FAILURE_CELLS}

    assert cells == {
        ("2x", 128, "calibrated-no-pins"),
        ("2x", 128, "calibrated+pins"),
        ("2x", 128, "hierarchical+pins-no-score"),
        ("4x", 1024, "fixed"),
        ("4x", 1024, "fixed+pins"),
        ("4x", 1024, "shuffled-quota"),
        ("4x", 1024, "shuffled-quota+pins"),
    }
    assert {(path.name, path.chunk_size, path.tiered) for path in diagnostic.DIAGNOSTIC_PATHS} == {
        ("resident-tokenwise", 1, False),
        ("tiered-tokenwise", 1, True),
        ("resident-chunk2", 2, False),
        ("tiered-chunk2", 2, True),
    }
    assert {name for name, _left, _right in diagnostic.PAIR_COMPARISONS} == {
        "A_tokenwise_resident_vs_tiered",
        "B_resident_chunk1_vs_chunk2",
        "D_tiered_chunk1_vs_chunk2",
        "E_chunk2_resident_vs_tiered",
        "F_original_resident_chunk2_vs_tiered_tokenwise",
    }
    args = diagnostic.build_parser().parse_args(["--output", str(tmp_path / "out.json")])
    assert args.repeats == 3
    assert "does not replace or relax" in diagnostic.DIAGNOSTIC_CLAIM_BOUNDARY


@pytest.mark.parametrize("value", ["1", "0", "-2", "2", "4"])
def test_parser_rejects_non_frozen_repeat_count(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(
            ["--output", str(tmp_path / "out.json"), "--repeats", value]
        )


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "-inf"])
def test_parser_rejects_invalid_tolerance(tmp_path: Path, value: str) -> None:
    with pytest.raises(SystemExit):
        diagnostic.build_parser().parse_args(
            ["--output", str(tmp_path / "out.json"), "--atol", value]
        )


def test_output_is_immutable_and_confined_to_diagnostic_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "diagnostic-root"
    monkeypatch.setattr(diagnostic, "DIAGNOSTIC_OUTPUT_ROOT", output_root)
    with pytest.raises(ValueError, match="primary equivalence root"):
        diagnostic._validate_output_path(
            diagnostic.PRIMARY_EQUIVALENCE_ROOT / "s55" / "diagnostic.json"
        )

    with pytest.raises(ValueError, match="frozen root"):
        diagnostic._validate_output_path(tmp_path / "outside.json")

    allowed = output_root / "s55" / "path-split-v1.raw.json"
    diagnostic._validate_output_path(allowed)
    allowed.parent.mkdir(parents=True)
    existing = allowed
    existing.write_text("preserve me")
    with pytest.raises(FileExistsError, match="already exists"):
        diagnostic._validate_output_path(existing)


def test_exclusive_json_publish_never_replaces_a_racing_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "diagnostic-root"
    monkeypatch.setattr(diagnostic, "DIAGNOSTIC_OUTPUT_ROOT", output_root)
    output = output_root / "s55" / "path-split-v1.raw.json"

    diagnostic._write_json_exclusive(output, {"status": "first"})
    assert json.loads(output.read_text()) == {"status": "first"}

    with pytest.raises(FileExistsError, match="already exists"):
        diagnostic._write_json_exclusive(output, {"status": "replacement"})
    assert json.loads(output.read_text()) == {"status": "first"}


def test_action_digest_is_finalize_order_neutral_and_locates_first_drift() -> None:
    actions = [
        _action(layer=4, batch=1, query=16),
        _action(layer=2, batch=0, query=8),
        _action(layer=4, batch=0, query=8),
    ]
    reordered = list(reversed(actions))

    assert diagnostic._json_digest(diagnostic._semantic_actions(actions)) == (
        diagnostic._json_digest(diagnostic._semantic_actions(reordered))
    )
    assert diagnostic._first_semantic_action_divergence(actions, reordered) is None

    drifted = [
        (
            {**action, "selected_end_positions": [3, 9]}
            if action["layer_index"] == 2
            else dict(action)
        )
        for action in reordered
    ]
    divergence = diagnostic._first_semantic_action_divergence(actions, drifted)

    assert divergence is not None
    assert divergence["identity"] == {
        "batch_index": 0,
        "query_position": 8,
        "layer_index": 2,
    }
    assert divergence["kind"] == "field_difference"
    assert divergence["differing_fields"] == ["selected_end_positions"]


def test_action_divergence_rejects_duplicate_causal_identity() -> None:
    duplicate = [_action(layer=2), _action(layer=2, selected=(9,))]

    with pytest.raises(ValueError, match="Duplicate semantic controller action identity"):
        diagnostic._first_semantic_action_divergence(duplicate, [])


def test_comparison_records_prediction_logit_and_first_action_drift() -> None:
    left = _success_observation("resident-tokenwise")
    right_logits = torch.tensor([[[2.0, 1.0], [0.5, 3.25]]])
    right = _success_observation(
        "tiered-tokenwise",
        predictions=[[1, 0]],
        logits=right_logits,
        actions=[_action(layer=4), _action(layer=2, selected=(3, 9))],
    )

    result = diagnostic._comparison(left, right, atol=0.1, rtol=0.0)

    assert result["comparable"] is True
    assert result["predictions_identical"] is False
    assert result["prediction_mismatch_count"] == 1
    assert result["prediction_mismatches"][0] == {
        "batch_index": 0,
        "conversation_id": "conversation:0",
        "query_index": 1,
        "query_input_position": 9,
        "left_prediction": 2,
        "right_prediction": 0,
        "left_margin": 2.5,
        "right_margin": 2.5,
    }
    assert result["path_neutral_actions_identical"] is False
    assert result["selected_positions_identical"] is False
    assert result["pin_sets_identical"] is True
    assert result["fallback_actions_identical"] is True
    assert result["first_semantic_action_divergence"]["identity"] == {
        "batch_index": 0,
        "query_position": 8,
        "layer_index": 2,
    }
    assert result["query_logits_bitwise_identical"] is False
    assert result["query_logits_allclose"] is False
    assert result["max_abs_logit_difference"] == pytest.approx(0.25)
    assert result["first_prediction_divergence"]["query_input_position"] == 9
    assert result["first_post_prefix_trace_divergence"] == {
        "batch_index": 0,
        "conversation_id": "conversation:0",
        "trace_token_index": 1,
        "input_position": 9,
        "predicted_token_position": 10,
        "logits_identical": False,
        "top1_identical": True,
        "left_top1": 1,
        "right_top1": 1,
        "left_logit_sha256": diagnostic._tensor_digest(
            torch.tensor([0.5, 3.0], dtype=torch.float32)
        ),
        "right_logit_sha256": diagnostic._tensor_digest(right_logits[0, 1]),
    }


def test_error_outcomes_are_reported_without_claiming_comparability() -> None:
    left = _error_observation("tiered-chunk2", repeat=0)
    same_error = _error_observation("tiered-chunk2", repeat=1)
    success = _success_observation("resident-chunk2")

    repeat_result = diagnostic._comparison(left, same_error, atol=0.0, rtol=0.0)
    mixed_result = diagnostic._comparison(left, success, atol=0.0, rtol=0.0)

    assert repeat_result["comparable"] is False
    assert repeat_result["error_outcomes_identical"] is True
    assert mixed_result["comparable"] is False
    assert mixed_result["error_outcomes_identical"] is False


def test_summary_discloses_unobserved_tiered_chunk2_interaction() -> None:
    runs: dict[str, list[diagnostic.RunObservation]] = {}
    for path in diagnostic.DIAGNOSTIC_PATHS:
        runs[path.name] = [
            (
                _error_observation(path.name, repeat=repeat)
                if path.name == "tiered-chunk2"
                else _success_observation(path.name, repeat=repeat)
            )
            for repeat in range(3)
        ]
    cell = {
        "arm": "calibrated+pins",
        "repeat_path_orders": [
            [path.name for path in diagnostic.DIAGNOSTIC_PATHS[index:]]
            + [path.name for path in diagnostic.DIAGNOSTIC_PATHS[:index]]
            for index in range(3)
        ],
        "runs": {
            name: [observation.payload for observation in observations]
            for name, observations in runs.items()
        },
        "comparisons": diagnostic._build_comparisons(runs, atol=0.0, rtol=0.0),
    }

    summary = diagnostic._summarize([cell], repeats=3)

    assert summary["expected_path_runs"] == 12
    assert summary["successful_path_runs"] == 9
    assert summary["error_path_runs"] == 3
    assert summary["full_2x2_interaction_observed"] is False
    assert summary["tiered_chunk2_observation"] == {
        "attempted_runs": 3,
        "successful_runs": 0,
        "error_runs": 3,
        "all_tiered_chunk2_runs_observed": False,
        "limitation_if_not_observed": (
            "The current tiered store unions selected blocks across the chunk query axis at "
            "a tokenwise-matched hot budget; an over-budget union is recorded as unsupported "
            "rather than interpreted as equivalence evidence."
        ),
    }
    assert (
        summary["paired_comparison_counts"]["D_tiered_chunk1_vs_chunk2"][
            "unsupported_or_error_observations"
        ]
        == 3
    )


def test_global_corner_order_is_deterministic_and_near_balanced() -> None:
    cells: list[dict[str, Any]] = []
    for cell_index in range(7):
        runs = {
            path.name: [_success_observation(path.name, repeat=repeat) for repeat in range(3)]
            for path in diagnostic.DIAGNOSTIC_PATHS
        }
        orders: list[list[str]] = []
        for repeat in range(3):
            rotation = (cell_index * 3 + repeat) % len(diagnostic.DIAGNOSTIC_PATHS)
            ordered = (
                *diagnostic.DIAGNOSTIC_PATHS[rotation:],
                *diagnostic.DIAGNOSTIC_PATHS[:rotation],
            )
            orders.append([path.name for path in ordered])
        cells.append(
            {
                "arm": f"diagnostic-arm-{cell_index}",
                "repeat_path_orders": orders,
                "runs": {
                    name: [observation.payload for observation in observations]
                    for name, observations in runs.items()
                },
                "comparisons": diagnostic._build_comparisons(runs, atol=0.0, rtol=0.0),
            }
        )

    summary = diagnostic._summarize(cells, repeats=3)

    assert summary["execution_order_balance"]["near_balanced_counts_differ_by_at_most_one"] is True
    assert set(
        count
        for position in summary["execution_order_balance"]["position_counts"].values()
        for count in position.values()
    ) == {5, 6}
    assert summary["full_2x2_interaction_observed"] is True


def test_logical_counter_projection_excludes_nondeterministic_timings() -> None:
    left = {
        "selected_queries": 8,
        "finalized_control_points": 4,
        "fallback_control_points": 0,
        "controller_time_ns": 100,
        "telemetry_time_ns": 20,
        "peak_selected_blocks": 12,
        "replay_digest": "a" * 64,
    }
    right = {**left, "controller_time_ns": 999, "telemetry_time_ns": 777}

    assert diagnostic._logical_controller_counters(left) == (
        diagnostic._logical_controller_counters(right)
    )


def test_frozen_manifest_binds_inputs_counts_and_targeted_statistics() -> None:
    manifest = diagnostic._load_frozen_manifest()
    path_split = manifest["path_split_diagnostic"]
    prospective = path_split["prospective_equivalence_panel"]
    targeted = manifest["targeted_causal_study"]

    assert path_split["localization_panel"]["expected_required_arm_batch_path_runs"] == 84
    assert path_split["localization_panel"]["expected_required_conversation_path_runs"] == 336
    assert prospective["expected_stage_a_total_path_runs"] == 9000
    assert prospective["expected_stage_b_total_path_runs"] == 4500
    assert prospective["maximum_expected_path_runs"] == 13500
    assert targeted["frozen_grid"]["paired_coordinates"] == 900
    assert targeted["statistics"]["conversation_outcome"].startswith("query accuracy")
    assert targeted["statistics"]["bootstrap_seed_by_contrast"] == {
        "stage_a_clean_layer_identity": 9171803,
        "stage_b_deployed_operational": 9171804,
    }
    assert path_split["artifact_contract"]["existing_artifact_overwrite_allowed"] is False


def test_tensor_digest_binds_dtype_shape_and_values() -> None:
    values = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    assert diagnostic._tensor_digest(values) == diagnostic._tensor_digest(values.clone())
    assert diagnostic._tensor_digest(values) != diagnostic._tensor_digest(values.double())
    assert diagnostic._tensor_digest(values) != diagnostic._tensor_digest(values.reshape(2, 1))
    assert diagnostic._tensor_digest(values) != diagnostic._tensor_digest(values + 1.0)
