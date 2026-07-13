from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    generate_associative_recall_batch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _latency_summary(latencies_ms: list[float], batch_size: int) -> dict[str, float]:
    elapsed = sum(latencies_ms) / 1000.0
    return {
        "p50_ms": _percentile(latencies_ms, 0.50),
        "p95_ms": _percentile(latencies_ms, 0.95),
        "p99_ms": _percentile(latencies_ms, 0.99),
        "mean_ms": sum(latencies_ms) / len(latencies_ms),
        "throughput_tokens_per_second": batch_size * len(latencies_ms) / elapsed,
    }


def _load_model(path: Path, device: torch.device) -> DeepSeekV4ForCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = DeepSeekV4Config(**payload["config"])
    model = DeepSeekV4ForCausalLM(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _set_topk(model: DeepSeekV4ForCausalLM, topk: int) -> None:
    for layer in model.model.layers:
        if layer.self_attn.csa is not None:
            layer.self_attn.csa.indexer.index_topk = topk


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.inference_mode()
def _prefill(
    model: DeepSeekV4ForCausalLM,
    prompt: torch.Tensor,
    tier_budget: int | None,
) -> tuple[object, float]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = model(prompt, use_cache=True)
    cache = output.past_key_values
    if cache is None:
        raise RuntimeError("Model did not return a cache.")
    if tier_budget is not None:
        cache.enable_csa_tiering(tier_budget)
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    del output
    return cache, ttft_ms


@torch.inference_mode()
def _decode(
    model: DeepSeekV4ForCausalLM,
    cache: object,
    tokens: torch.Tensor,
) -> tuple[list[torch.Tensor], list[float]]:
    logits: list[torch.Tensor] = []
    latencies: list[float] = []
    for index in range(tokens.shape[1]):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output = model(
            tokens[:, index : index + 1],
            past_key_values=cache,
            use_cache=True,
        )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
        logits.append(output.logits.detach().cpu())
    return logits, latencies


def _cache_compressor_bytes(cache: object) -> int:
    total = 0
    for layer in cache.layers:
        values = layer.compressed_kv.get("compressor")
        positions = layer.compressed_positions.get("compressor")
        for tensor in (values, positions):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return int(total)


@torch.inference_mode()
def benchmark_scenario(
    model: DeepSeekV4ForCausalLM,
    *,
    name: str,
    context_length: int,
    batch_size: int,
    decode_tokens: int,
    topk: int,
    seed: int,
    device: torch.device,
) -> dict:
    _set_topk(model, topk)
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, context_length),
        generator=generator,
    ).to(device)
    tokens = torch.randint(
        0,
        model.config.vocab_size,
        (batch_size, decode_tokens),
        generator=generator,
    ).to(device)
    block_count = context_length // model.config.compress_rates["compressed_sparse_attention"]
    tier_budget = min(block_count, topk * batch_size)

    _cleanup()
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident, resident_ttft = _prefill(model, prompt, None)
    resident_allocated = torch.cuda.memory_allocated(device)
    resident_reserved = torch.cuda.memory_reserved(device)
    resident_peak = torch.cuda.max_memory_allocated(device)
    resident_compressor = _cache_compressor_bytes(resident)
    resident_logits, resident_latencies = _decode(model, resident, tokens)
    del resident
    _cleanup()

    tier_baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    tiered, tiered_ttft = _prefill(model, prompt, tier_budget)
    tiered_allocated = torch.cuda.memory_allocated(device)
    tiered_reserved = torch.cuda.memory_reserved(device)
    tiered_peak = torch.cuda.max_memory_allocated(device)
    tiered_logits, tiered_latencies = _decode(model, tiered, tokens)
    tier_stats = tiered.tiered_memory_stats()

    resident_tensor = torch.cat(resident_logits, dim=1).float()
    tiered_tensor = torch.cat(tiered_logits, dim=1).float()
    exact = bool(torch.equal(resident_tensor, tiered_tensor))
    max_error = float((resident_tensor - tiered_tensor).abs().max())
    resident_delta = resident_allocated - baseline_allocated
    tiered_delta = tiered_allocated - tier_baseline_allocated
    result = {
        "name": name,
        "context_length": context_length,
        "batch_size": batch_size,
        "decode_tokens": decode_tokens,
        "topk": topk,
        "tier_hot_budget_blocks_per_layer": tier_budget,
        "correctness": {
            "logits_bitwise_equal": exact,
            "logits_close_bfloat16": bool(
                torch.allclose(resident_tensor, tiered_tensor, atol=0.125, rtol=0.01)
            ),
            "max_absolute_logit_error": max_error,
            "greedy_tokens_equal": bool(
                torch.equal(resident_tensor.argmax(-1), tiered_tensor.argmax(-1))
            ),
        },
        "resident": {
            "ttft_ms": resident_ttft,
            "cache_allocated_bytes": resident_delta,
            "allocated_bytes_after_prefill": resident_allocated,
            "reserved_bytes_after_prefill": resident_reserved,
            "fragmentation_bytes_after_prefill": resident_reserved - resident_allocated,
            "fragmentation_ratio_after_prefill": (resident_reserved - resident_allocated)
            / max(resident_reserved, 1),
            "peak_allocated_bytes": resident_peak,
            "compressor_value_bytes": resident_compressor,
            "decode": _latency_summary(resident_latencies, batch_size),
        },
        "tiered": {
            "ttft_ms": tiered_ttft,
            "cache_allocated_bytes": tiered_delta,
            "allocated_bytes_after_prefill": tiered_allocated,
            "reserved_bytes_after_prefill": tiered_reserved,
            "fragmentation_bytes_after_prefill": tiered_reserved - tiered_allocated,
            "fragmentation_ratio_after_prefill": (tiered_reserved - tiered_allocated)
            / max(tiered_reserved, 1),
            "peak_allocated_bytes": tiered_peak,
            "decode": _latency_summary(tiered_latencies, batch_size),
            "store": {
                "logical_blocks": sum(item.logical_blocks for item in tier_stats),
                "hot_blocks": sum(item.hot_blocks for item in tier_stats),
                "logical_bytes": sum(item.logical_bytes for item in tier_stats),
                "hot_bytes": sum(item.hot_bytes for item in tier_stats),
                "host_bytes": sum(item.host_bytes for item in tier_stats),
                "h2d_bytes": sum(item.h2d_bytes for item in tier_stats),
                "d2h_bytes": sum(item.d2h_bytes for item in tier_stats),
                "h2d_count": sum(item.h2d_count for item in tier_stats),
                "d2h_count": sum(item.d2h_count for item in tier_stats),
                "useful_h2d_bytes": sum(item.useful_h2d_bytes for item in tier_stats),
                "late_misses": sum(item.late_misses for item in tier_stats),
                "prefetches": sum(item.prefetches for item in tier_stats),
                "evictions": sum(item.evictions for item in tier_stats),
            },
        },
    }
    result["comparison"] = {
        "cache_allocated_reduction_bytes": resident_delta - tiered_delta,
        "cache_allocated_reduction_ratio": (resident_delta - tiered_delta)
        / max(resident_delta, 1),
        "ttft_ratio": tiered_ttft / resident_ttft,
        "decode_p95_ratio": result["tiered"]["decode"]["p95_ms"]
        / result["resident"]["decode"]["p95_ms"],
        "throughput_ratio": result["tiered"]["decode"]["throughput_tokens_per_second"]
        / result["resident"]["decode"]["throughput_tokens_per_second"],
        "mean_runtime_overhead_ms_per_token": result["tiered"]["decode"]["mean_ms"]
        - result["resident"]["decode"]["mean_ms"],
    }
    del tiered
    _cleanup()
    return result


@torch.inference_mode()
def benchmark_concurrent_requests(
    model: DeepSeekV4ForCausalLM,
    *,
    requests: int,
    context_length: int,
    decode_tokens: int,
    seed: int,
    device: torch.device,
) -> dict:
    """Round-robin independent request caches on one CUDA execution stream."""

    topk = model.config.index_topk
    _set_topk(model, topk)
    generator = torch.Generator().manual_seed(seed)
    prompts = torch.randint(
        0, model.config.vocab_size, (requests, context_length), generator=generator
    ).to(device)
    tokens = torch.randint(
        0, model.config.vocab_size, (requests, decode_tokens), generator=generator
    ).to(device)

    def build(tiered: bool) -> tuple[list[object], float]:
        caches = []
        started = time.perf_counter_ns()
        for request in range(requests):
            cache, _ = _prefill(
                model,
                prompts[request : request + 1],
                topk if tiered else None,
            )
            caches.append(cache)
        torch.cuda.synchronize()
        return caches, (time.perf_counter_ns() - started) / 1_000_000.0

    def decode(caches: list[object]) -> tuple[list[torch.Tensor], list[float]]:
        logits = []
        latencies = []
        for token_index in range(decode_tokens):
            for request in range(requests):
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                output = model(
                    tokens[request : request + 1, token_index : token_index + 1],
                    past_key_values=caches[request],
                    use_cache=True,
                )
                torch.cuda.synchronize()
                latencies.append((time.perf_counter_ns() - started) / 1_000_000.0)
                logits.append(output.logits.detach().cpu())
        return logits, latencies

    _cleanup()
    resident_base = torch.cuda.memory_allocated(device)
    resident, resident_ttft = build(False)
    resident_allocated = torch.cuda.memory_allocated(device)
    resident_logits, resident_latencies = decode(resident)
    del resident
    _cleanup()

    tiered_base = torch.cuda.memory_allocated(device)
    tiered, tiered_ttft = build(True)
    tiered_allocated = torch.cuda.memory_allocated(device)
    tiered_logits, tiered_latencies = decode(tiered)
    stores = [item for cache in tiered for item in cache.tiered_memory_stats()]
    resident_tensor = torch.cat(resident_logits, dim=0).float()
    tiered_tensor = torch.cat(tiered_logits, dim=0).float()
    resident_decode = _latency_summary(resident_latencies, 1)
    tiered_decode = _latency_summary(tiered_latencies, 1)
    result = {
        "name": "concurrent-r2",
        "execution": "round-robin independent caches on one CUDA stream",
        "requests": requests,
        "context_length": context_length,
        "decode_tokens_per_request": decode_tokens,
        "correctness": {
            "logits_bitwise_equal": bool(torch.equal(resident_tensor, tiered_tensor)),
            "logits_close_bfloat16": bool(
                torch.allclose(resident_tensor, tiered_tensor, atol=0.125, rtol=0.01)
            ),
            "greedy_tokens_equal": bool(
                torch.equal(resident_tensor.argmax(-1), tiered_tensor.argmax(-1))
            ),
            "max_absolute_logit_error": float(
                (resident_tensor - tiered_tensor).abs().max()
            ),
        },
        "resident": {
            "aggregate_ttft_ms": resident_ttft,
            "cache_allocated_bytes": resident_allocated - resident_base,
            "decode": resident_decode,
        },
        "tiered": {
            "aggregate_ttft_ms": tiered_ttft,
            "cache_allocated_bytes": tiered_allocated - tiered_base,
            "decode": tiered_decode,
            "store": {
                "host_bytes": sum(item.host_bytes for item in stores),
                "hot_bytes": sum(item.hot_bytes for item in stores),
                "h2d_bytes": sum(item.h2d_bytes for item in stores),
                "d2h_bytes": sum(item.d2h_bytes for item in stores),
                "prefetches": sum(item.prefetches for item in stores),
                "evictions": sum(item.evictions for item in stores),
                "late_misses": sum(item.late_misses for item in stores),
            },
        },
        "comparison": {
            "cache_allocated_reduction_bytes": (resident_allocated - resident_base)
            - (tiered_allocated - tiered_base),
            "decode_p95_ratio": tiered_decode["p95_ms"] / resident_decode["p95_ms"],
            "throughput_ratio": tiered_decode["throughput_tokens_per_second"]
            / resident_decode["throughput_tokens_per_second"],
        },
    }
    del tiered
    _cleanup()
    return result


@torch.inference_mode()
def quality_check(
    model: DeepSeekV4ForCausalLM,
    *,
    examples: int,
    seed: int,
    device: torch.device,
) -> dict:
    _set_topk(model, model.config.index_topk)
    task = AssociativeRecallConfig(
        vocab_size=model.config.vocab_size,
        sliding_window=model.config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    generator = torch.Generator().manual_seed(seed)
    resident_correct = 0
    tiered_correct = 0
    equal = 0
    maximum_error = 0.0
    for _ in range(examples):
        batch = generate_associative_recall_batch(
            task,
            batch_size=1,
            sequence_length=80,
            generator=generator,
            device=device,
        )
        resident, _ = _prefill(model, batch.input_ids[:, :-1], None)
        resident_output = model(
            batch.input_ids[:, -1:], past_key_values=resident, use_cache=True
        )
        tiered, _ = _prefill(model, batch.input_ids[:, :-1], model.config.index_topk)
        tiered_output = model(batch.input_ids[:, -1:], past_key_values=tiered, use_cache=True)
        resident_logits = resident_output.logits[:, -1]
        tiered_logits = tiered_output.logits[:, -1]
        resident_correct += int(resident_logits.argmax(-1).eq(batch.targets).sum())
        tiered_correct += int(tiered_logits.argmax(-1).eq(batch.targets).sum())
        equal += int(torch.equal(resident_logits, tiered_logits))
        maximum_error = max(maximum_error, float((resident_logits - tiered_logits).abs().max()))
    return {
        "examples": examples,
        "resident_accuracy": resident_correct / examples,
        "tiered_accuracy": tiered_correct / examples,
        "bitwise_equal_examples": equal,
        "max_absolute_logit_error": maximum_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Adaptive V4 Memory M4 tiering.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--quality-examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=6071401)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M4 runtime benchmark requires CUDA.")
    device = torch.device("cuda")
    model = _load_model(args.checkpoint, device)
    native_topk = model.config.index_topk
    warmup_prompt = torch.zeros((1, 512), dtype=torch.long, device=device)
    warmup_token = torch.zeros((1, 1), dtype=torch.long, device=device)
    warmup_cache, _ = _prefill(model, warmup_prompt, native_topk)
    model(warmup_token, past_key_values=warmup_cache, use_cache=True)
    del warmup_cache, warmup_prompt, warmup_token
    _cleanup()
    scenarios = [
        ("steady-b1", 512, 1, native_topk),
        ("batch-b4", 512, 4, native_topk),
        ("query-shift-b1", 1024, 1, native_topk),
        ("dense-fallback-b1", 512, 1, 512 // 4),
    ]
    results = [
        benchmark_scenario(
            model,
            name=name,
            context_length=context,
            batch_size=batch,
            decode_tokens=args.decode_tokens,
            topk=topk,
            seed=args.seed + index,
            device=device,
        )
        for index, (name, context, batch, topk) in enumerate(scenarios)
    ]
    results.append(
        benchmark_concurrent_requests(
            model,
            requests=2,
            context_length=512,
            decode_tokens=args.decode_tokens,
            seed=args.seed + 100,
            device=device,
        )
    )
    _set_topk(model, native_topk)
    quality = quality_check(
        model,
        examples=args.quality_examples,
        seed=args.seed + 1000,
        device=device,
    )
    payload = {
        "schema_version": 1,
        "experiment_id": "m4-tiered-runtime-v1",
        "scale": args.scale,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "seed": args.seed,
        "scenarios": results,
        "quality": quality,
        "limitations": [
            "Reference PyTorch gather path; no fused paged-attention kernel.",
            "CSA compressor values are tiered; index vectors and rollback history stay on GPU.",
            "Single accelerator and synthetic associative-recall quality task.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
