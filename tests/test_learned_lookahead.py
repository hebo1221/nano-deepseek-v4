from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    RISK_FEATURE_NAMES,
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    LearnedLookaheadPolicy,
    OnlineTrainingFreeController,
    RankedBlock,
    ReplayQuery,
    RiskExample,
    TrainingFreeControllerConfig,
    load_deepseek_v4_cache,
    save_deepseek_v4_cache,
    train_learned_lookahead_policy,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import collect_p1_online_lookahead_labels as labels  # noqa: E402
import evaluate_p1_online_learned_lookahead_shard as lookahead_eval  # noqa: E402
import run_p1_online_learned_lookahead as lookahead_matrix  # noqa: E402
import summarize_p1_online_learned_lookahead as lookahead_summary  # noqa: E402


def _examples(prefix: str, count: int, *, dense: bool = True) -> tuple[RiskExample, ...]:
    return tuple(
        RiskExample(
            group_id=f"{prefix}-{index}",
            features=tuple(
                float(((index + 1) * (feature + 3)) % 17) / 17.0
                for feature in range(len(RISK_FEATURE_NAMES))
            ),
            sufficient_topk=1 + index % 2,
            dense_required=dense and index % 5 == 0,
        )
        for index in range(count)
    )


def _policy(*, layers: int = 2, dense: bool = True) -> LearnedLookaheadPolicy:
    return train_learned_lookahead_policy(
        _examples("train", 24, dense=dense),
        _examples("calibration", 12, dense=dense),
        csa_layer_count=layers,
        normal_global_budget=2 * layers,
        dense_global_budget=4 * layers,
        hidden_size=4,
        steps=40,
        seed=17,
    )


def _controller_config(layers: int = 2) -> TrainingFreeControllerConfig:
    return TrainingFreeControllerConfig(
        global_block_budget=2 * layers,
        dense_fallback_block_budget=4 * layers,
        top_p=0.8,
        min_blocks_per_layer=1,
        max_extra_blocks_per_layer=0,
        uncertainty_threshold=1.0,
        dense_cardinality_threshold=1.0,
        enable_dense_fallback=False,
    )


def _observe(controller: OnlineTrainingFreeController, layer: int) -> None:
    controller.observe(
        layer_index=layer,
        query_positions=torch.tensor([[15]]),
        block_end_positions=torch.tensor([[3, 7, 11, 15]]),
        scores=torch.tensor([[[8.0, 2.0, 1.0, 0.5]]]),
        native_mask=torch.tensor([[[True, True, False, False]]]),
        block_bytes=64,
    )


def _probe_query(layer: int, position: int = 9) -> ReplayQuery:
    return ReplayQuery(
        trace_id="label-test",
        request_id="request-0",
        layer_index=layer,
        batch_index=0,
        query_position=position,
        phase="decode",
        logical_block_count=2,
        block_bytes=64,
        native_block_ids=(f"l{layer}:b0:e3",),
        ranked_blocks=(
            RankedBlock(block_id=f"l{layer}:b0:e3", score=4.0),
            RankedBlock(block_id=f"l{layer}:b0:e7", score=1.0),
        ),
    )


def test_learned_lookahead_training_is_split_safe_frozen_and_deterministic() -> None:
    first = _policy()
    second = _policy()

    assert first == second
    assert first.hidden_size == 4
    assert len(first.policy_digest) == 64
    prediction = first.predict(_examples("test", 1)[0].features)
    assert prediction.calibrated_topk in (1, 2)
    assert prediction.active_global_budget <= first.dense_global_budget
    assert LearnedLookaheadPolicy.from_dict(json.loads(json.dumps(first.to_dict()))) == first

    with pytest.raises(ValueError, match="overlap"):
        train_learned_lookahead_policy(
            _examples("shared", 4),
            _examples("shared", 4),
            csa_layer_count=2,
            normal_global_budget=4,
            dense_global_budget=8,
            steps=1,
        )


def test_online_learned_lookahead_protocol_freezes_full_separate_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            root / "research/adaptive_v4_memory/manifests/p1-online-learned-lookahead-v1.json"
        ).read_text()
    )

    assert manifest["status"] == "frozen_before_execution"
    assert manifest["role"].startswith("separate exploratory")
    assert manifest["causal_contract"]["applied_time"] == "token t+1 only"
    assert manifest["causal_contract"]["bootstrap"].startswith("native selection")
    assert manifest["splits"]["training_model_seeds"] == [
        6071401,
        6071402,
        6071403,
        6071404,
        6071405,
    ]
    matrix = manifest["matrix"]
    assert len(matrix["scales"]) == 2
    assert len(matrix["workload_families"]) == 9
    assert len(matrix["contexts"]) == 5
    assert matrix["test_examples_per_family_seed_scale_budget"] == 1000
    assert matrix["test_examples_per_arm"] == 180_000
    assert "online-learned-lookahead+pins" in manifest["arms"]
    assert "minimum attainable p=0.0625" in manifest["positive_gate"]["five_seed_test_resolution"]
    assert manifest["claim_boundary"]["legacy_m3"].endswith("no online-lookahead evidence")
    assert lookahead_matrix.EXPECTED_LABEL_SHARDS == 6_750
    assert lookahead_matrix.EXPECTED_POLICIES == 20
    assert lookahead_matrix.EXPECTED_TEST_SHARDS == 9_000
    assert tuple(manifest["arms"]) == lookahead_eval.ARMS
    for name in ("matrix_runner", "parallel_matrix_runner", "summary_runner"):
        assert (root / manifest["implementation"][name]).is_file()
    assert manifest["execution_equivalence"]["required_scale_seed_probes"] == 10


def test_online_matrix_rejects_stale_implementation_and_dependency(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    artifact = tmp_path / "label.json"
    payload = {
        "experiment_id": "label-v1",
        "scale": "s55",
        "source": {"dirty": False, "implementation_digest": "implementation"},
        "design": {
            "path": str(lookahead_matrix.DESIGN),
            "sha256": lookahead_matrix.sha256(lookahead_matrix.DESIGN),
        },
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": lookahead_matrix.sha256(checkpoint),
        },
    }
    artifact.write_text(json.dumps(payload))

    assert lookahead_matrix._valid(
        artifact,
        "label-v1",
        {"scale": "s55"},
        implementation_digest="implementation",
    )
    lookahead_summary._verify_common(
        payload,
        implementation_digest="implementation",
        label="test label",
    )
    assert not lookahead_matrix._valid(
        artifact,
        "label-v1",
        {"scale": "s55"},
        implementation_digest="stale",
    )
    checkpoint.write_bytes(b"checkpoint-v2")
    assert not lookahead_matrix._valid(
        artifact,
        "label-v1",
        {"scale": "s55"},
        implementation_digest="implementation",
    )


def test_label_builder_uses_prior_token_and_dense_prediction_not_target() -> None:
    predictions: dict[int | str, torch.Tensor] = {
        1: torch.tensor([[7]]),
        2: torch.tensor([[7]]),
        4: torch.tensor([[3]]),
        8: torch.tensor([[4]]),
        "dense": torch.tensor([[7]]),
    }
    rows, failures = labels.build_label_rows(
        queries=(_probe_query(2), _probe_query(5)),
        predictions=predictions,
        query_columns={10: 0},
        conversation_ids=("conversation-0",),
        input_ids=torch.arange(16).unsqueeze(0),
        split="train",
        scale="s55",
        training_seed=6071401,
        family="single-remote-retrieval",
        context=80,
        replicate=0,
        dense_topk=20,
    )

    assert failures == []
    assert rows[0]["feature_token_position"] == 9
    assert rows[0]["label_token_position"] == 10
    assert rows[0]["causal_offset"] == 1
    assert rows[0]["sufficient_topk"] == 1
    assert rows[0]["dense_required"] is False

    predictions[1] = predictions[2] = predictions[4] = predictions[8] = torch.tensor([[0]])
    dense_rows, _ = labels.build_label_rows(
        queries=(_probe_query(2), _probe_query(5)),
        predictions=predictions,
        query_columns={10: 0},
        conversation_ids=("conversation-0",),
        input_ids=torch.arange(16).unsqueeze(0),
        split="calibration",
        scale="s55",
        training_seed=6071401,
        family="single-remote-retrieval",
        context=80,
        replicate=0,
        dense_topk=20,
    )
    assert dense_rows[0]["sufficient_topk"] == 20
    assert dense_rows[0]["dense_required"] is True


def test_label_generation_seed_namespaces_are_disjoint() -> None:
    common = {
        "training_seed": 6071401,
        "family": "single-remote-retrieval",
        "context": 80,
        "replicate": 0,
    }
    assert labels.generation_seed(split="train", **common) != labels.generation_seed(
        split="calibration", **common
    )
    assert lookahead_eval.generation_seed(**common) not in {
        labels.generation_seed(split="train", **common),
        labels.generation_seed(split="calibration", **common),
    }


def _label_audit_payload(checkpoint: Path) -> dict[str, object]:
    scale = "s55"
    training_seed = labels.TRAINING_SEEDS[0]
    split = "train"
    family = labels.PAPER_GRADE_WORKLOAD_FAMILIES[0]
    context = labels.CONTEXTS[0]
    replicate = labels.REPLICATES[split][0]
    rows = []
    for index in range(labels.EXAMPLES_PER_SHARD):
        conversation_id = f"{family}:{context}:{index}"
        label_position = 12
        rows.append(
            {
                "group_id": (
                    f"{split}:{scale}:{training_seed}:{family}:{context}:{replicate}:"
                    f"{conversation_id}:q{label_position}"
                ),
                "features": [0.25] * len(RISK_FEATURE_NAMES),
                "sufficient_topk": 1,
                "dense_required": False,
                "conversation_id": conversation_id,
                "input_sha256": hashlib.sha256(conversation_id.encode()).hexdigest(),
                "feature_token_position": label_position - 1,
                "label_token_position": label_position,
                "dense_prediction": 7,
                "matching_registered_topk": [1, 2],
                "causal_offset": 1,
            }
        )
    return {
        "experiment_id": "p1-online-learned-lookahead-label-shard-v1",
        "scale": scale,
        "training_seed": training_seed,
        "split": split,
        "family": family,
        "context": context,
        "replicate": replicate,
        "generation_seed": labels.generation_seed(
            split=split,
            training_seed=training_seed,
            family=family,
            context=context,
            replicate=replicate,
        ),
        "conversations": labels.EXAMPLES_PER_SHARD,
        "risk_examples": len(rows),
        "failures": [],
        "failure_count": 0,
        "registered_topk_sweep": list(labels.NORMAL_TOPK_SWEEP),
        "dense_topk": [20],
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": lookahead_summary.sha256(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "rows_digest": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
        "leakage_guard": {
            "features_use_prior_token_only": True,
            "causal_offset": 1,
            "benchmark_targets_used_for_labels": False,
            "dense_model_predictions_used_for_labels": True,
            "split_namespace": labels.SPLIT_NAMESPACES[split],
        },
    }


def _test_audit_payload(checkpoint: Path) -> dict[str, object]:
    scale = "s55"
    training_seed = labels.TRAINING_SEEDS[0]
    budget = "2x"
    family = labels.PAPER_GRADE_WORKLOAD_FAMILIES[0]
    context = labels.CONTEXTS[0]
    replicate = 0
    records = []
    for arm in lookahead_eval.ARMS:
        for index in range(labels.EXAMPLES_PER_SHARD):
            conversation_id = f"{family}:{context}:{index}"
            records.append(
                {
                    "arm": arm,
                    "conversation_id": conversation_id,
                    "batch_offset": (index // labels.BATCH_SIZE) * labels.BATCH_SIZE,
                    "input_sha256": hashlib.sha256(conversation_id.encode()).hexdigest(),
                    "family": family,
                    "context": context,
                    "replicate": replicate,
                    "budget": budget,
                    "targets": [3, 5],
                    "predictions": [3, 5],
                    "correct": [True, True],
                    "correct_count": 2,
                    "total": 2,
                    "controller": (
                        None
                        if arm == "native-resident"
                        else {
                            "control_points": 1,
                            "selected_blocks_sum": 1,
                            "budget_blocks_sum": 2,
                            "budget_violations": 0,
                        }
                    ),
                    "failure": None,
                }
            )
    records.sort(key=lambda row: (row["arm"], row["conversation_id"]))
    accounting = {
        "state_bytes": 1,
        "hca_bytes": 2,
        "csa_bytes": 3,
        "index_bytes": 4,
        "logical_cache_bytes": 10,
        "hot_resident_bytes": 6,
        "cold_resident_bytes": 4,
    }
    metrics = []
    for offset in range(0, labels.EXAMPLES_PER_SHARD, labels.BATCH_SIZE):
        rotation = (replicate * 5 + offset // labels.BATCH_SIZE) % len(lookahead_eval.ARMS)
        order = (*lookahead_eval.ARMS[rotation:], *lookahead_eval.ARMS[:rotation])
        for execution_index, arm in enumerate(order):
            metrics.append(
                {
                    "batch_offset": offset,
                    "arm": arm,
                    "execution_index": execution_index,
                    "wall_ms": 1.5,
                    "peak_cuda_allocated_bytes": 100,
                    "peak_cuda_reserved_bytes": 120,
                    "accounting": accounting,
                    "tier": {
                        "hot_blocks": 0,
                        "logical_blocks": 0,
                        "hot_bytes": 0,
                        "host_bytes": 0,
                        "h2d_bytes": 0,
                        "useful_h2d_bytes": 0,
                        "d2h_bytes": 0,
                        "late_misses": 0,
                        "evictions": 0,
                    },
                    "failure": None,
                    "physical_hot_budget_blocks_by_layer": (
                        None if arm == "native-resident" else {"0": 4}
                    ),
                    "budget_violations": 0,
                }
            )
    return {
        "experiment_id": "p1-online-learned-lookahead-test-shard-v1",
        "scale": scale,
        "training_seed": training_seed,
        "budget": budget,
        "family": family,
        "context": context,
        "replicate": replicate,
        "generation_seed": lookahead_eval.generation_seed(
            training_seed=training_seed,
            family=family,
            context=context,
            replicate=replicate,
        ),
        "examples": labels.EXAMPLES_PER_SHARD,
        "arms": list(lookahead_eval.ARMS),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": lookahead_summary.sha256(checkpoint),
        },
        "records_digest": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "records": records,
        "batch_metrics": metrics,
    }


def test_summary_recomputes_label_seed_coordinates_and_row_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _label_audit_payload(checkpoint)
    coordinate, inputs, digest = lookahead_summary._validate_label_payload(
        payload, path=tmp_path / "label.json"
    )
    assert coordinate == (
        "s55",
        labels.TRAINING_SEEDS[0],
        "train",
        labels.PAPER_GRADE_WORKLOAD_FAMILIES[0],
        labels.CONTEXTS[0],
        0,
    )
    assert len(inputs) == labels.EXAMPLES_PER_SHARD
    assert digest == lookahead_summary.sha256(checkpoint)

    payload["generation_seed"] = int(payload["generation_seed"]) + 1
    with pytest.raises(ValueError, match="label shard accounting"):
        lookahead_summary._validate_label_payload(payload, path=tmp_path / "label.json")


def test_summary_recomputes_test_pairing_physical_schema_and_aggregates(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = _test_audit_payload(checkpoint)
    result = lookahead_summary._validate_test_payload(payload, path=tmp_path / "test.json")
    assert len(result[2]) == labels.EXAMPLES_PER_SHARD
    assert result[4:6] == (0, 0)

    paired_tamper = json.loads(json.dumps(payload))
    paired_tamper["records"][0]["input_sha256"] = "0" * 64
    paired_tamper["records_digest"] = hashlib.sha256(
        json.dumps(paired_tamper["records"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="Paired test inputs"):
        lookahead_summary._validate_test_payload(
            paired_tamper, path=tmp_path / "paired-tamper.json"
        )

    physical_tamper = json.loads(json.dumps(payload))
    physical_tamper["batch_metrics"][0]["wall_ms"] = float("nan")
    with pytest.raises(ValueError, match="physical metric schema"):
        lookahead_summary._validate_test_payload(
            physical_tamper, path=tmp_path / "physical-tamper.json"
        )

    aggregate_tamper = json.loads(json.dumps(payload))
    aggregate_tamper["batch_metrics"][0]["budget_violations"] = 1
    with pytest.raises(ValueError, match="budget violation aggregate"):
        lookahead_summary._validate_test_payload(
            aggregate_tamper, path=tmp_path / "aggregate-tamper.json"
        )


def test_online_matrix_requires_terminal_digest_bound_p2_gate(tmp_path: Path) -> None:
    path = tmp_path / "causal.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": "p2-causal-ablation-audit-v1",
                "audit": {
                    "unique_shards": 9000,
                    "all_raw_shards_verified": True,
                    "all_dependency_digests_verified": True,
                    "all_record_digests_verified": True,
                    "no_budget_violations": True,
                },
            }
        )
    )
    assert lookahead_matrix.require_causal_gate(path)["audit"]["unique_shards"] == 9000
    path.write_text(json.dumps({"experiment_id": "p2-causal-ablation-audit-v1", "audit": {}}))
    with pytest.raises(RuntimeError, match="terminal digest-bound P2 causal audit"):
        lookahead_matrix.require_causal_gate(path)


def test_online_learned_lookahead_is_causal_bounded_and_digest_replayable() -> None:
    policy = _policy()
    controller = OnlineTrainingFreeController(
        _controller_config(),
        (2, 5),
        protected_end_positions=(3,),
        learned_policy=policy,
    )
    native = torch.ones(1, 1, 4, dtype=torch.bool)
    assert torch.equal(
        controller.apply(
            layer_index=2,
            query_positions=torch.tensor([[15]]),
            block_end_positions=torch.tensor([[3, 7, 11, 15]]),
            native_mask=native,
        ),
        native,
    )
    _observe(controller, 2)
    _observe(controller, 5)
    controller.finalize()

    assert controller.controller_kind == "learned-lookahead"
    assert controller.stats().native_bootstrap_queries == 1
    assert len(controller.last_actions) == 1
    action = controller.last_actions[0]
    assert action.query_position == 15
    assert action.selected_blocks <= action.budget_limit <= policy.dense_global_budget
    assert all(layer.pinned_block_ids for layer in action.layers)
    assert controller.replay_actions() == controller.last_actions

    payload = json.loads(json.dumps(controller.to_dict()))
    restored = OnlineTrainingFreeController.from_dict(payload)
    assert restored.to_dict() == controller.to_dict()
    payload["counters"]["replay_digest"] = "0" * 64
    with pytest.raises(ValueError, match="replay digest"):
        OnlineTrainingFreeController.from_dict(payload)

    ablated = OnlineTrainingFreeController(
        _controller_config(),
        (2, 5),
        learned_policy=policy,
        enable_learned_dense_fallback=False,
    )
    _observe(ablated, 2)
    _observe(ablated, 5)
    ablated.finalize()
    assert ablated.enable_learned_dense_fallback is False
    assert all(action.fallback_reason is None for action in ablated.replay_actions())
    assert (
        OnlineTrainingFreeController.from_dict(
            json.loads(json.dumps(ablated.to_dict()))
        ).enable_learned_dense_fallback
        is False
    )


def test_learned_lookahead_cache_lifecycle_and_physical_tiering(tmp_path: Path) -> None:
    torch.manual_seed(41)
    config = DeepSeekV4Config(num_nextn_predict_layers=0)
    model = DeepSeekV4ForCausalLM(config).eval()
    assert config.layer_types is not None
    csa_layers = sum(
        layer_type == "compressed_sparse_attention" for layer_type in config.layer_types
    )
    policy = _policy(layers=csa_layers, dense=False)
    controller_config = _controller_config(csa_layers)
    cache = DeepSeekV4Cache(config)
    cache.enable_learned_lookahead_controller(controller_config, policy)
    prompt = torch.randint(0, config.vocab_size, (1, 32))
    model(prompt, past_key_values=cache, use_cache=True)
    assert cache.online_memory_controller is not None
    assert cache.online_memory_controller.controller_kind == "learned-lookahead"

    tiered = cache.clone()
    tiered.enable_csa_tiering(hot_budget_blocks=2)
    model(torch.tensor([[23]]), past_key_values=tiered, use_cache=True)
    assert all(stats.hot_blocks <= 2 for stats in tiered.tiered_memory_stats())

    cache_dir = tmp_path / "learned-lookahead-cache"
    save_deepseek_v4_cache(tiered, cache_dir)
    restored = load_deepseek_v4_cache(config, cache_dir)
    assert restored.online_memory_controller is not None
    assert restored.online_memory_controller.to_dict() == tiered.online_memory_controller.to_dict()

    restored.crop(28, config)
    selected = restored.select_batch(0)
    stacked = DeepSeekV4Cache.stack([selected, selected.clone()])
    assert stacked.online_memory_controller is not None
    assert stacked.online_memory_controller.controller_kind == "learned-lookahead"
    output = model(torch.tensor([[29], [29]]), past_key_values=stacked, use_cache=True)
    assert output.logits.shape[:2] == (2, 1)
