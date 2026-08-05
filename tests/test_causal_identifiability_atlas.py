from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import torch

from nano_deepseek_v4 import CSASelectionProbe, DeepSeekV4Config, DeepSeekV4ForCausalLM

SCRIPTS = Path(__file__).parents[1] / "research" / "adaptive_v4_memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
atlas = importlib.import_module("run_causal_identifiability_atlas")
summary = importlib.import_module("summarize_causal_identifiability_atlas")


def _manifest() -> dict:
    return {
        "design": [
            {
                "scale": "s151",
                "training_seeds": list(range(5)),
                "contexts": [640, 1024],
                "families": list(atlas.FAMILIES),
                "replicates": [0, 1, 2],
                "budget_multiplier": 1,
            },
            {
                "scale": "s55",
                "training_seeds": list(range(5)),
                "contexts": [640],
                "families": list(atlas.FAMILIES),
                "replicates": [0, 1],
                "budget_multiplier": 2,
            },
        ]
    }


def _model() -> DeepSeekV4ForCausalLM:
    return DeepSeekV4ForCausalLM(
        DeepSeekV4Config(
            vocab_size=64,
            hidden_size=16,
            moe_intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=8,
            q_lora_rank=8,
            n_routed_experts=4,
            num_experts_per_tok=2,
            num_hash_layers=0,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            sliding_window=8,
            o_groups=2,
            o_lora_rank=4,
            index_n_heads=2,
            index_head_dim=4,
            index_topk=2,
            num_nextn_predict_layers=0,
            layer_types=["compressed_sparse_attention"],
        )
    ).eval()


def test_design_has_160_cells_and_60_exhaustive_cells() -> None:
    coordinates = atlas.coordinates(_manifest())

    assert len(coordinates) == 160
    assert sum(atlas.is_exhaustive_coordinate(item) for item in coordinates) == 60


def test_workload_seed_excludes_checkpoint_and_scale() -> None:
    first = {
        "scale": "s151",
        "training_seed": 6071406,
        "budget_multiplier": 1,
        "context": 640,
        "family": "single-remote-retrieval",
        "replicate": 2,
    }
    second = {**first, "scale": "s55", "training_seed": 6071410, "budget_multiplier": 2}

    assert atlas.workload_identity(first) == atlas.workload_identity(second)
    assert atlas.evaluation_seed(first, "namespace") == atlas.evaluation_seed(
        second, "namespace"
    )
    assert atlas.evaluation_seed(first, "namespace") != atlas.evaluation_seed(
        {**first, "replicate": 1}, "namespace"
    )


def test_observable_routes_are_finite_cardinality_and_evidence_free() -> None:
    torch.manual_seed(7)
    model = _model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 64))
    probe = CSASelectionProbe()
    with torch.no_grad():
        model(tokens, use_cache=False, csa_probe=probe)

    identity, policies, receipts, digest = atlas.build_observable_routes(
        model,
        probe,
        query_positions=torch.tensor([[63]]),
        topk=2,
        trace_id="test",
        workload={"context": 64, "family": "single-remote-retrieval", "replicate": 0},
        capture_candidates=True,
    )

    assert len(identity.selections) == 1
    assert set(policies) == set(atlas.OBSERVABLE_POLICIES)
    assert len(digest) == 64
    assert "evidence" not in atlas.build_observable_routes.__annotations__
    for policy in policies.values():
        assert len(policy.selections) == 1
        assert len(policy.selections[0].block_end_positions) == 2
    assert len(receipts[0]["candidates"]) == receipts[0]["eligible_blocks"]
    proxy = [
        row["scores"]["perturbation-proxy"] for row in receipts[0]["candidates"]
    ]
    assert all(math.isfinite(value) and value >= 0.0 for value in proxy)


def test_batch_plan_preserves_empty_and_variable_routes() -> None:
    plan = atlas._batch_plan(
        trace_id="test",
        request_id="batch",
        layer_index=2,
        query_position=63,
        selections=[(), (3,), (3, 7, 11)],
    )

    assert [row.block_end_positions for row in plan.selections] == [(), (3,), (3, 7, 11)]
    assert [row.batch_index for row in plan.selections] == [0, 1, 2]


def test_final_candidate_chunk_is_padded_to_the_frozen_batch_shape() -> None:
    assert atlas._pad_candidate_chunk((3, 7, 11), 8) == (
        3,
        7,
        11,
        11,
        11,
        11,
        11,
        11,
    )


def test_spearman_uses_average_tie_ranks() -> None:
    assert summary.spearman([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == 1.0
    assert summary.spearman([1.0, 1.0, 1.0], [4.0, 5.0, 6.0]) == 0.0
    assert math.isclose(
        summary.spearman([1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 2.0, 4.0]),
        0.8333333333333335,
    )


def test_checkpoint_cluster_bootstrap_is_deterministic() -> None:
    first = summary._bootstrap_ratio(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 4.0, 6.0, 8.0, 10.0],
        draws=1_000,
        seed=17,
    )
    second = summary._bootstrap_ratio(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 4.0, 6.0, 8.0, 10.0],
        draws=1_000,
        seed=17,
    )

    assert first == second == (0.5, 0.5)
