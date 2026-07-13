from __future__ import annotations

import pytest
import torch

from nano_deepseek_v4 import (
    CSAProbeObjective,
    CSASelectionProbe,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    evidence_block_indices,
)


def _probe_model() -> DeepSeekV4ForCausalLM:
    config = DeepSeekV4Config(
        vocab_size=128,
        hidden_size=32,
        moe_intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        sliding_window=8,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        num_nextn_predict_layers=0,
        layer_types=["compressed_sparse_attention"] * 2,
    )
    return DeepSeekV4ForCausalLM(config)


def test_evidence_positions_map_to_first_containing_block():
    ends = torch.tensor([[3, 7, 11], [3, 7, 11]])
    evidence = torch.tensor([[0, 3, 4, 11], [1, 7, 8, 10]])

    indices = evidence_block_indices(ends, evidence)

    assert indices.tolist() == [[0, 0, 1, 2], [0, 1, 2, 2]]
    with pytest.raises(ValueError, match="not represented"):
        evidence_block_indices(ends, torch.tensor([[12], [1]]))


def test_probe_is_opt_in_and_does_not_change_model_output():
    torch.manual_seed(7)
    model = _probe_model().eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 24))

    expected, _, _ = model.model(input_ids)
    probe = CSASelectionProbe()
    actual, _, _ = model.model(input_ids, csa_probe=probe)

    assert torch.equal(actual, expected)
    assert len(probe.records) == 2
    assert [record.layer_index for record in probe.records] == [0, 1]
    for record in probe.records:
        assert record.scores.shape == (2, 24, 6)
        assert record.value_blocks.shape == (2, 6, model.config.head_dim)
        assert record.scores.requires_grad
        assert record.value_blocks.requires_grad


def test_probe_objective_backpropagates_to_indexer_and_compressor():
    torch.manual_seed(11)
    model = _probe_model().train()
    objective = CSAProbeObjective(
        head_dim=model.config.head_dim,
        query_dim=model.config.q_lora_rank,
        vocab_size=model.config.vocab_size,
    )
    input_ids = torch.randint(0, model.config.vocab_size, (2, 24))
    query_positions = torch.tensor([[20, 21], [20, 21]])
    evidence_positions = torch.tensor([[1, 9], [5, 13]])
    targets = torch.tensor([[17, 18], [19, 20]])
    probe = CSASelectionProbe()

    model.model(input_ids, csa_probe=probe)
    losses = objective(
        probe.records,
        query_positions=query_positions,
        evidence_positions=evidence_positions,
        targets=targets,
    )
    (losses.ranking + losses.read_ranking + losses.value).backward()

    first_csa = model.model.layers[0].self_attn.csa
    assert first_csa is not None
    assert losses.layers == 2
    assert losses.examples == 8
    assert torch.isfinite(losses.ranking)
    assert torch.isfinite(losses.read_ranking)
    assert torch.isfinite(losses.value)
    assert 0.0 <= losses.ranking_accuracy <= 1.0
    assert 0.0 <= losses.read_ranking_accuracy <= 1.0
    assert 0.0 <= losses.value_accuracy <= 1.0
    assert first_csa.indexer.q_b_proj.weight.grad is not None
    assert first_csa.indexer.q_b_proj.weight.grad.abs().sum() > 0
    assert first_csa.kv_proj.weight.grad is not None
    assert first_csa.kv_proj.weight.grad.abs().sum() > 0


def test_probe_rejects_cached_forward():
    model = _probe_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

    with pytest.raises(ValueError, match="without a cache"):
        model(input_ids, use_cache=True, csa_probe=CSASelectionProbe())
