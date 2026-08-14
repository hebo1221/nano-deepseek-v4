from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import tour

ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def report() -> tour.TourReport:
    return tour.run_tour()


def test_tour_executes_every_tiny_architecture_family(report: tour.TourReport):
    assert report.schema_version == 1
    assert report.fixture == "tiny-architecture-tour-v1"
    assert report.device == "cpu"
    assert report.input_ids == tuple(range(8))
    assert report.input_shape == (1, 8)
    assert report.embedding_shape == (1, 8, 64)
    assert report.initial_streams_shape == (1, 8, 4, 64)
    assert report.final_collapsed_shape == (1, 8, 64)
    assert report.final_hidden_shape == (1, 8, 64)
    assert report.logits_shape == (1, 8, 512)

    assert [layer.attention_kind for layer in report.layers] == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert [layer.mlp_kind for layer in report.layers] == [
        "hash_moe",
        "hash_moe",
        "hash_moe",
        "moe",
    ]
    for layer in report.layers:
        assert layer.stream_input_shape == (1, 8, 4, 64)
        assert layer.attention_input_shape == (1, 8, 64)
        assert layer.attention_output_shape == (1, 8, 64)
        assert layer.moe_output_shape == (1, 8, 64)
        assert layer.router_logits_shape == (1, 8, 8)
        assert layer.stream_output_shape == (1, 8, 4, 64)

    csa = report.layers[2]
    assert csa.compression_rate == 2
    assert csa.compressed_slot_count == 4
    assert csa.selected_slots_per_query == (0, 2)
    hca = report.layers[3]
    assert hca.compression_rate == 4
    assert hca.compressed_slot_count == 2
    assert hca.selected_slots_per_query is None

    assert len(report.mtp) == 1
    assert report.mtp[0].previous_streams_shape == (1, 7, 4, 64)
    assert report.mtp[0].future_embeddings_shape == (1, 7, 64)
    assert report.mtp[0].output_streams_shape == (1, 7, 4, 64)
    assert report.mtp[0].hidden_shape == (1, 7, 64)
    assert report.mtp[0].logits_shape == (1, 7, 512)

    assert report.prefix_cache_tokens == 5
    assert report.final_cache_tokens == 8
    assert report.cache_matches is True
    assert report.passed is True
    assert "not evidence" in report.claim_boundary.lower()


def test_tour_preserves_the_callers_torch_rng_state():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()

    tour.run_tour(seed=17)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_tour_does_not_seed_accelerator_generators(
    monkeypatch: pytest.MonkeyPatch,
):
    accelerator_seeds: list[int] = []
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: accelerator_seeds.append(seed),
    )

    tour.run_tour(seed=17)

    assert accelerator_seeds == []


def test_tour_json_is_structured_and_cli_safe(
    report: tour.TourReport,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(tour, "run_tour", lambda **kwargs: report)

    assert tour.main(["--seed", "99", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == 1
    assert payload["layers"][2]["attention_kind"] == "compressed_sparse_attention"
    assert payload["mtp"][0]["logits_shape"] == [1, 7, 512]
    assert payload["cache_matches"] is True


def test_human_tour_points_from_execution_to_the_reading_guide(
    report: tour.TourReport,
    capsys: pytest.CaptureFixture[str],
):
    tour._print_human(report)

    output = capsys.readouterr().out
    assert "mHC expands one hidden state into 4 streams" in output
    assert "L2: compressed_sparse + hash_moe" in output
    assert "L3: heavily_compressed + moe" in output
    assert "cached/full match: True" in output
    assert (
        "https://github.com/hebo1221/nano-deepseek-v4/blob/"
        "main/docs/guides/modeling-walkthrough.md"
    ) in output


def test_packaged_tour_points_to_its_versioned_walkthrough(tmp_path: Path):
    assert tour._walkthrough_url(source_root=tmp_path) == (
        "https://github.com/hebo1221/nano-deepseek-v4/blob/"
        f"v{tour.__version__}/docs/guides/modeling-walkthrough.md"
    )


def test_public_docs_link_the_executable_walkthrough():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "guides" / "modeling-walkthrough.md").read_text(
        encoding="utf-8"
    )

    for document in (readme, docs_index, guide):
        assert "nano-deepseek-v4 tour" in document
    for symbol in (
        "HyperConnection",
        "DeepSeekV4Attention",
        "DeepSeekV4MoE",
        "DeepSeekV4MTPModule",
    ):
        assert symbol in guide
