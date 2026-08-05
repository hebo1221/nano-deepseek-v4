from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

from nano_deepseek_v4 import CSASelectionProbe, DeepSeekV4Config, DeepSeekV4ForCausalLM

SCRIPTS = Path(__file__).parents[1] / "research" / "adaptive_v4_memory" / "scripts"
sys.path.insert(0, str(SCRIPTS))
runner = importlib.import_module("run_restoration_ceiling_e2a")
summary = importlib.import_module("summarize_restoration_ceiling_e2a")


def _manifest() -> dict:
    return {
        "design": [
            {
                "scale": "s151",
                "training_seeds": list(range(5)),
                "contexts": [640, 1024],
                "families": list(runner.FAMILIES),
                "replicates": [0, 1, 2],
                "budget_multiplier": 1,
            },
            {
                "scale": "s55",
                "training_seeds": list(range(5)),
                "contexts": [640],
                "families": list(runner.FAMILIES),
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
            index_topk=4,
            num_nextn_predict_layers=0,
            layer_types=["compressed_sparse_attention"],
        )
    ).eval()


def test_design_has_exactly_160_cells() -> None:
    assert len(runner.coordinates(_manifest())) == 160


def test_workload_seed_excludes_checkpoint_and_scale() -> None:
    first = {
        "scale": "s151",
        "training_seed": 6071406,
        "budget_multiplier": 1,
        "context": 640,
        "family": "single-remote-retrieval",
        "replicate": 2,
    }
    second = {**first, "scale": "s55", "training_seed": 6071410}

    assert runner.workload_identity(first) == runner.workload_identity(second)
    assert runner.evaluation_seed(first, "e2a") == runner.evaluation_seed(second, "e2a")
    assert runner.evaluation_seed(first, "e2a") != runner.evaluation_seed(first, "e1")


def test_observable_routes_have_exact_absolute_k_and_full_coverage() -> None:
    torch.manual_seed(17)
    model = _model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 64))
    probe = CSASelectionProbe()
    with torch.no_grad():
        model(tokens, use_cache=False, csa_probe=probe)

    identity, policies, receipts, digest = runner.build_observable_routes(
        probe,
        query_positions=torch.tensor([[63]]),
        native_topk=1,
        trace_id="test",
    )

    assert len(identity.selections) == 1
    assert set(policies) == set(runner.OBSERVABLE_ARMS)
    assert len(digest) == 64
    selected = receipts[0]["selected_block_ends"]
    assert len(selected["native"]) == 1
    for topk in runner.ABSOLUTE_INDEX_K:
        assert len(selected[f"index-k{topk}"]) == topk
        assert len(set(selected[f"index-k{topk}"])) == topk
    assert len(selected["full-compressed"]) == receipts[0]["eligible_blocks"]
    assert "evidence" not in runner.build_observable_routes.__annotations__


def test_native_equivalent_plan_matches_requested_cardinality() -> None:
    torch.manual_seed(23)
    model = _model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 64))
    probe = CSASelectionProbe()
    with torch.no_grad():
        native = model(tokens, use_cache=False, csa_probe=probe)
        identity, policies, _, _ = runner.build_observable_routes(
            probe,
            query_positions=torch.tensor([[63]]),
            native_topk=4,
            trace_id="test",
        )
        identity_output = model(tokens, use_cache=False, selection_plan=identity)
        absolute_output = model(tokens, use_cache=False, selection_plan=policies["index-k4"])

    assert torch.equal(native.logits, identity_output.logits)
    assert torch.equal(native.logits, absolute_output.logits)


def test_checkpoint_bootstraps_are_deterministic() -> None:
    assert summary._bootstrap_mean([1, 2, 3, 4, 5], draws=2_000, seed=11) == (
        summary._bootstrap_mean([1, 2, 3, 4, 5], draws=2_000, seed=11)
    )
    assert summary._bootstrap_ratio(
        [1, 2, 3, 4, 5], [2, 4, 6, 8, 10], draws=2_000, seed=13
    ) == (0.5, 0.5)


def test_frozen_decision_tree_does_not_promote_a_failed_teacher() -> None:
    gates = {
        "s151_640_full_teacher": False,
        "s151_1024_length_shift": True,
        "distributed_beyond_k8": True,
        "evidence_headroom_recovery": True,
        "s55_2x_safety": True,
    }
    assert summary._decision(gates) == (
        "NO_GO_FULL_CACHE_TEACHER_PIVOT_EXTERNAL_RETRIEVAL"
    )
    assert summary._decision({name: True for name in gates}) == (
        "GO_SEPARATELY_FROZEN_COMPACT_RESTORATION_EXPERIMENT"
    )
