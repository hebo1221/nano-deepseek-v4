from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import CSASelectionProbe, DeepSeekV4Config, DeepSeekV4ForCausalLM

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from hca_csa_h0_probe import (  # noqa: E402
    HCACSAReadSidecar,
    extract_hca_csa_feature_rows,
    hca_csa_feature_rows_digest,
)


def _hca_csa_model() -> DeepSeekV4ForCausalLM:
    return DeepSeekV4ForCausalLM(
        DeepSeekV4Config(
            vocab_size=64,
            hidden_size=16,
            moe_intermediate_size=24,
            num_hidden_layers=3,
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
            layer_types=[
                "compressed_sparse_attention",
                "heavily_compressed_attention",
                "compressed_sparse_attention",
            ],
        )
    )


def test_research_sidecar_is_output_neutral_and_restores_model() -> None:
    torch.manual_seed(6071401)
    model = _hca_csa_model().eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 256))

    with torch.no_grad():
        expected, _, _ = model.model(input_ids)
        probe = CSASelectionProbe()
        with HCACSAReadSidecar(model) as sidecar:
            actual, _, _ = model.model(input_ids, csa_probe=probe)

    assert torch.equal(actual, expected)
    assert [(read.layer_index, read.memory_type) for read in sidecar.reads] == [
        (0, "csa"),
        (1, "hca"),
        (2, "csa"),
    ]
    hca_read = sidecar.reads[1]
    assert hca_read.entry_end_positions.tolist() == [[127, 255]]
    assert hca_read.head_scores.shape == (1, 2, 256, 2)
    assert torch.isfinite(hca_read.head_probabilities).all()
    assert torch.all(hca_read.head_probabilities >= 0)

    rows = extract_hca_csa_feature_rows(
        probe,
        sidecar.reads,
        trace_id="sidecar-regression",
        query_positions=torch.tensor([[200, 255]]),
        native_topk=model.config.index_topk,
        hca_rate=model.config.compress_rates["heavily_compressed_attention"],
    )
    assert rows
    assert all(row.prior_csa_layer < row.hca_layer < row.target_csa_layer for row in rows)
    assert hca_csa_feature_rows_digest(rows) == hca_csa_feature_rows_digest(rows)

    for layer in model.model.layers:
        assert "_core_attention" not in layer.self_attn.__dict__
    with torch.no_grad():
        restored, _, _ = model.model(input_ids)
    assert torch.equal(restored, expected)


def test_research_sidecar_rejects_training_mode() -> None:
    with pytest.raises(ValueError, match="requires eval mode"):
        HCACSAReadSidecar(_hca_csa_model().train())


def test_research_sidecar_restores_partial_installation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _hca_csa_model().eval()
    sidecar = HCACSAReadSidecar(model)
    second_compressor = model.model.layers[1].self_attn.hca
    assert second_compressor is not None

    def fail_registration(_hook: object) -> object:
        raise RuntimeError("injected hook failure")

    monkeypatch.setattr(second_compressor, "register_forward_hook", fail_registration)
    with pytest.raises(RuntimeError, match="injected hook failure"):
        sidecar.__enter__()

    assert not sidecar._active
    assert not sidecar._handles
    assert not sidecar._restores
    for layer in model.model.layers:
        assert "_core_attention" not in layer.self_attn.__dict__
