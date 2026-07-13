from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from nano_deepseek_v4 import (
    AdaptiveMemoryTraceCollector,
    CacheAdvanceEvent,
    CSASelectionEvent,
    DeepSeekV4Cache,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    MemoryTraceConfig,
    load_memory_trace,
    measure_cache_memory,
    replay_native_selected_sets,
)

_CACHE_TENSOR_ATTRIBUTES = ("local_kv", "local_positions")
_CACHE_DICTIONARY_ATTRIBUTES = (
    "buffer_kv",
    "buffer_gate",
    "buffer_positions",
    "history_kv",
    "history_gate",
    "history_positions",
    "compressed_kv",
    "compressed_positions",
    "overlap_kv",
    "overlap_gate",
    "overlap_positions",
)


def _tiny_model() -> DeepSeekV4ForCausalLM:
    torch.manual_seed(0)
    model = DeepSeekV4ForCausalLM(DeepSeekV4Config())
    model.eval()
    return model


def _assert_caches_equal(left: DeepSeekV4Cache, right: DeepSeekV4Cache) -> None:
    assert left.seen_tokens == right.seen_tokens
    assert left.config == right.config
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        for attribute in _CACHE_TENSOR_ATTRIBUTES:
            left_tensor = getattr(left_layer, attribute)
            right_tensor = getattr(right_layer, attribute)
            if left_tensor is None or right_tensor is None:
                assert left_tensor is right_tensor
            else:
                assert torch.equal(left_tensor, right_tensor)
        for attribute in _CACHE_DICTIONARY_ATTRIBUTES:
            left_values = getattr(left_layer, attribute)
            right_values = getattr(right_layer, attribute)
            assert left_values.keys() == right_values.keys()
            for name in left_values:
                left_tensor = left_values[name]
                right_tensor = right_values[name]
                if left_tensor is None or right_tensor is None:
                    assert left_tensor is right_tensor
                else:
                    assert torch.equal(left_tensor, right_tensor)


def _selection_map(collector: AdaptiveMemoryTraceCollector):
    return {
        (item.layer_index, item.batch_index, item.query_position): item.block_ids
        for item in replay_native_selected_sets(collector.result())
    }


def test_trace_on_off_preserves_logits_cache_and_torch_rng_state():
    model = _tiny_model()
    ids = torch.tensor([[41, 7, 89, 13, 5, 144, 8, 55, 3, 233, 21, 34]])
    initial_rng = torch.random.get_rng_state()

    untraced = model(ids, use_cache=True)
    untraced_rng = torch.random.get_rng_state()

    torch.random.set_rng_state(initial_rng)
    collector = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="invariance"))
    traced = model(ids, use_cache=True, memory_trace=collector)
    traced_rng = torch.random.get_rng_state()

    assert torch.equal(traced.logits, untraced.logits)
    assert torch.equal(traced_rng, untraced_rng)
    assert traced.past_key_values is not None
    assert untraced.past_key_values is not None
    _assert_caches_equal(traced.past_key_values, untraced.past_key_values)

    assert any(isinstance(event, CSASelectionEvent) for event in collector.events)
    advances = [event for event in collector.events if isinstance(event, CacheAdvanceEvent)]
    assert len(advances) == 1
    accounting = advances[0].accounting
    assert accounting.logical_cache_bytes == (
        accounting.state_bytes
        + accounting.hca_bytes
        + accounting.csa_bytes
        + accounting.index_bytes
    )
    assert accounting.hot_resident_bytes == accounting.logical_cache_bytes
    assert accounting.cold_resident_bytes == 0
    assert advances[0].host_to_device_bytes == 0
    assert advances[0].device_to_host_bytes == 0


def test_trace_preserves_greedy_generation():
    model = _tiny_model()
    ids = torch.tensor([[17, 9, 4, 31, 62, 7]])
    expected = model.generate(ids, max_new_tokens=4)
    collector = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="generation"))
    actual = model.generate(ids, max_new_tokens=4, memory_trace=collector)

    assert torch.equal(actual, expected)
    advances = [event for event in collector.events if isinstance(event, CacheAdvanceEvent)]
    assert advances[0].phase == "prefill"
    assert all(event.phase == "decode" for event in advances[1:])
    assert advances[-1].seen_tokens_after == ids.shape[1] + actual.shape[1] - ids.shape[1] - 1


def test_trace_jsonl_digest_privacy_and_replay(tmp_path: Path):
    model = _tiny_model()
    ids = torch.tensor([[12, 44, 7, 98, 6, 3, 77, 19]])
    collector = AdaptiveMemoryTraceCollector(
        MemoryTraceConfig(trace_id="persisted", request_id="synthetic-request")
    )
    model(ids, use_cache=True, memory_trace=collector)

    written = collector.write(tmp_path / "trace")
    loaded = load_memory_trace(tmp_path / "trace")
    assert loaded == written
    assert loaded.manifest.contains_raw_text is False
    assert replay_native_selected_sets(loaded) == replay_native_selected_sets(tmp_path / "trace")

    jsonl = (tmp_path / "trace" / "events.jsonl").read_text(encoding="utf-8")
    for forbidden_key in ('"input_ids"', '"token_ids"', '"prompt"', '"text"'):
        assert forbidden_key not in jsonl

    event_lines = [json.loads(line) for line in jsonl.splitlines()]
    event_lines[0]["phase"] = "training"
    invalid_payload = "".join(
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
        for event in event_lines
    )
    (tmp_path / "trace" / "events.jsonl").write_text(invalid_payload, encoding="utf-8")
    manifest_path = tmp_path / "trace" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events_sha256"] = hashlib.sha256(invalid_payload.encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid memory trace event"):
        load_memory_trace(tmp_path / "trace")

    with (tmp_path / "trace" / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_memory_trace(tmp_path / "trace")


def test_stable_block_ids_match_full_and_chunked_cache_execution():
    model = _tiny_model()
    ids = torch.tensor([[2, 18, 5, 71, 9, 33, 6, 42, 15, 88, 11, 27]])

    full_collector = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="full"))
    full = model(ids, use_cache=True, memory_trace=full_collector)

    chunked_collector = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="chunked"))
    first = model(ids[:, :7], use_cache=True, memory_trace=chunked_collector)
    assert first.past_key_values is not None
    second = model(ids[:, 7:], past_key_values=first.past_key_values, use_cache=True)

    assert torch.allclose(
        torch.cat([first.logits, second.logits], dim=1),
        full.logits,
        atol=1e-4,
        rtol=1e-4,
    )
    assert _selection_map(chunked_collector) == _selection_map(full_collector)
    assert all(
        block_id.startswith(f"l{layer_index}:b{batch_index}:e")
        for (layer_index, batch_index, _), block_ids in _selection_map(full_collector).items()
        for block_id in block_ids
    )


def test_empty_cache_accounting_and_trace_api_validation():
    config = DeepSeekV4Config()
    accounting = measure_cache_memory(DeepSeekV4Cache(config))
    assert accounting.logical_cache_bytes == 0
    assert accounting.hot_resident_bytes == 0

    with pytest.raises(ValueError, match="trace_id"):
        MemoryTraceConfig(trace_id="")

    model = _tiny_model()
    collector = AdaptiveMemoryTraceCollector(MemoryTraceConfig(trace_id="invalid-use"))
    with pytest.raises(ValueError, match="requires use_cache"):
        model(torch.tensor([[1, 2]]), memory_trace=collector)
