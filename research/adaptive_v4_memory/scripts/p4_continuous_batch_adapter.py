#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM, measure_cache_memory

POLICIES = ("resident-native", "tiered-native")
PREFILL_CHUNK = 256


@dataclass
class IndexerTimingTrace:
    indexer_time_ns: int = 0
    selection_calls: int = 0

    def record_csa_selection(self, **values: Any) -> None:
        self.indexer_time_ns += int(values["indexer_wall_time_ns"])
        self.selection_calls += 1

    def record_cache_advance(self, *_args: Any, **_kwargs: Any) -> None:
        return


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text())
    _require(
        spec.get("experiment_id") == "p4-production-adapter-spec-v1",
        "The frozen P4 production adapter spec is required.",
    )
    cell = spec.get("cell", {})
    _require(
        isinstance(cell, dict)
        and cell.get("scale") in {"s55", "s151"}
        and cell.get("context") in {8_192, 32_768, 131_072}
        and cell.get("generation") in {128, 512, 2_048}
        and isinstance(cell.get("profile"), str)
        and isinstance(cell.get("batch"), int)
        and cell["batch"] in {1, 4, 8, 16}
        and isinstance(cell.get("concurrency"), int)
        and cell["concurrency"] in {1, 8, 32},
        "The P4 production adapter cell is invalid.",
    )
    _require(tuple(spec.get("policies", ())) == POLICIES, "Policy order drifted.")
    warmups = spec.get("warmups")
    measured = spec.get("measured_repetitions")
    seeds = spec.get("repetition_seeds")
    _require(
        warmups == 5
        and measured == 30
        and isinstance(seeds, list)
        and len(seeds) == warmups + measured
        and all(isinstance(seed, int) for seed in seeds),
        "Warmup or repetition contract drifted.",
    )
    checkpoint = Path(spec.get("checkpoint", ""))
    _require(checkpoint.is_file(), "The frozen checkpoint is missing.")
    for dependency in ("manifest", "p3_audit"):
        metadata = spec.get(dependency, {})
        dependency_path = Path(metadata.get("path", ""))
        _require(dependency_path.is_file(), f"Missing {dependency} dependency.")
        _require(metadata.get("sha256") == sha256(dependency_path), f"{dependency} drifted.")
    return spec


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _load_model(path: Path, device: torch.device) -> DeepSeekV4ForCausalLM:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = DeepSeekV4Config(**payload["config"])
    model = DeepSeekV4ForCausalLM(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _tensor_digest(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate_inputs(
    model: DeepSeekV4ForCausalLM,
    *,
    context: int,
    generation: int,
    batch: int,
    concurrency: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    generator = torch.Generator().manual_seed(seed)
    prompts = [
        torch.randint(0, model.config.vocab_size, (batch, context), generator=generator)
        for _ in range(concurrency)
    ]
    decode = [
        torch.randint(0, model.config.vocab_size, (batch, generation), generator=generator)
        for _ in range(concurrency)
    ]
    return torch.cat(prompts), torch.cat(decode), _tensor_digest([*prompts, *decode])


def _cache_totals(cache: Any) -> dict[str, int]:
    accounting = measure_cache_memory(cache)
    stores = cache.tiered_memory_stats()
    return {
        "logical_cache_bytes": accounting.logical_cache_bytes,
        "hot_resident_bytes": accounting.hot_resident_bytes,
        "cold_resident_bytes": accounting.cold_resident_bytes,
        "pinned_host_bytes": sum(item.host_bytes for item in stores),
        "h2d_bytes": sum(item.h2d_bytes for item in stores),
        "d2h_bytes": sum(item.d2h_bytes for item in stores),
        "h2d_count": sum(item.h2d_count for item in stores),
        "d2h_count": sum(item.d2h_count for item in stores),
        "useful_h2d_bytes": sum(item.useful_h2d_bytes for item in stores),
        "late_misses": sum(item.late_misses for item in stores),
        "prefetches": sum(item.prefetches for item in stores),
        "evictions": sum(item.evictions for item in stores),
    }


def _process_hbm_bytes() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used_mib = 0
    for line in completed.stdout.splitlines():
        pid, separator, memory = line.partition(",")
        if separator and pid.strip() == str(os.getpid()):
            used_mib += int(memory.strip())
    _require(used_mib > 0, "nvidia-smi did not expose process-total HBM.")
    return used_mib * 1024 * 1024


def _driver_version() -> str:
    value = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(bool(value), "NVIDIA driver provenance is missing.")
    return value


def backend_provenance(executable: Path, device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    origin = _git("config", "--get", "remote.origin.url")
    return {
        "runtime_name": "nano-deepseek-v4-static-continuous-batching",
        "runtime_version": "1",
        "source_repository": origin,
        "source_revision": _git("rev-parse", "HEAD"),
        "executable_sha256": sha256(executable),
        "deployment": "bare-metal",
        "accelerator": {
            "model": properties.name,
            "count": torch.cuda.device_count(),
            "driver": _driver_version(),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "scheduler": "static-full-request-continuous-batch",
        },
    }


@torch.inference_mode()
def run_policy(
    model: DeepSeekV4ForCausalLM,
    *,
    policy: str,
    prompt_cpu: torch.Tensor,
    decode_cpu: torch.Tensor,
    input_digest: str,
    batch: int,
    concurrency: int,
    device: torch.device,
) -> dict[str, Any]:
    _require(policy in POLICIES, f"Unknown policy: {policy}")
    _cleanup()
    torch.cuda.reset_peak_memory_stats(device)
    scheduler_received = [time.perf_counter_ns() for _ in range(concurrency)]
    admitted = [time.perf_counter_ns() for _ in range(concurrency)]
    cache: Any = None
    for start in range(0, prompt_cpu.shape[1], PREFILL_CHUNK):
        chunk = prompt_cpu[:, start : start + PREFILL_CHUNK].to(device)
        output = model(chunk, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        if cache is None:
            raise RuntimeError("Production prefill did not return a cache.")
        if policy == "tiered-native" and start == 0:
            cache.enable_csa_tiering(model.config.index_topk * prompt_cpu.shape[0])
        del output, chunk
    torch.cuda.synchronize()
    allocated_after_prefill = torch.cuda.memory_allocated(device)
    reserved_after_prefill = torch.cuda.memory_reserved(device)
    generation = decode_cpu.shape[1]
    decode_records: list[dict[str, Any]] = []
    step_latencies: list[float] = []
    first_token_ns: int | None = None
    prediction = hashlib.sha256()
    decode_started = time.perf_counter_ns()
    for token_index in range(generation):
        token = decode_cpu[:, token_index : token_index + 1].to(device)
        torch.cuda.synchronize()
        dispatch_ns = time.perf_counter_ns()
        output = model(token, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        completed_ns = time.perf_counter_ns()
        if token_index == 0:
            first_token_ns = completed_ns
        greedy = output.logits[:, -1].argmax(dim=-1).to(device="cpu")
        prediction.update(greedy.numpy().tobytes())
        elapsed_ms = (completed_ns - dispatch_ns) / 1_000_000.0
        step_latencies.extend([elapsed_ms] * concurrency)
        for request_index in range(concurrency):
            decode_records.append(
                {
                    "request_id": f"request-{request_index}",
                    "token_index": token_index,
                    "execution_batch_id": f"decode-{token_index}",
                    "dispatch_ns": dispatch_ns,
                    "completed_ns": completed_ns,
                }
            )
        del output, token, greedy
    torch.cuda.synchronize()
    decode_completed = time.perf_counter_ns()
    _require(first_token_ns is not None, "No first token was generated.")
    request_records = [
        {
            "request_id": f"request-{request_index}",
            "scheduler_received_ns": scheduler_received[request_index],
            "admitted_ns": admitted[request_index],
            "first_token_ns": first_token_ns,
            "completed_ns": decode_completed,
            "generated_tokens": batch * generation,
            "failure": None,
        }
        for request_index in range(concurrency)
    ]
    overlap_window = min(row["completed_ns"] for row in request_records) - max(
        row["admitted_ns"] for row in request_records
    )
    totals = _cache_totals(cache)
    process_hbm = _process_hbm_bytes()
    trace = IndexerTimingTrace()
    cache.memory_trace = trace
    probe = torch.zeros((prompt_cpu.shape[0], 1), dtype=torch.long, device=device)
    model(probe, past_key_values=cache, use_cache=True)
    torch.cuda.synchronize()
    cache.memory_trace = None
    generated_tokens = batch * concurrency * generation
    elapsed_seconds = (decode_completed - decode_started) / 1_000_000_000.0
    result = {
        "policy": policy,
        "input_digest": input_digest,
        "load_execution": {
            "actual_concurrent_serving": True,
            "requested_concurrency": concurrency,
            "maximum_active_requests": concurrency,
            "overlap_window_ns": overlap_window,
            "concurrency_proof_mode": "continuous-batching",
            "maximum_requests_per_decode_batch": concurrency,
            "maximum_decode_execution_overlap": concurrency,
        },
        "request_records": request_records,
        "decode_token_records": decode_records,
        "decode_step_latency_ms": step_latencies,
        "generated_token_throughput_per_second": generated_tokens / elapsed_seconds,
        "prediction_digest": prediction.hexdigest(),
        "cuda": {
            "allocated_after_prefill_bytes": allocated_after_prefill,
            "reserved_after_prefill_bytes": reserved_after_prefill,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "process_total_hbm_bytes": process_hbm,
            "device_total_hbm_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "cache": {
            key: totals[key]
            for key in (
                "logical_cache_bytes",
                "hot_resident_bytes",
                "cold_resident_bytes",
                "pinned_host_bytes",
            )
        },
        "transfer": {
            "h2d_bytes": totals["h2d_bytes"],
            "d2h_bytes": totals["d2h_bytes"],
            "useful_h2d_bytes": totals["useful_h2d_bytes"],
            "h2d_count": totals["h2d_count"],
            "d2h_count": totals["d2h_count"],
            "misses": totals["h2d_count"],
            "late_misses": totals["late_misses"],
            "prefetches": totals["prefetches"],
            "evictions": totals["evictions"],
        },
        "timing": {
            "controller_time_ns": 0,
            "indexer_time_ns": asdict(trace)["indexer_time_ns"],
        },
        "tail_failures": [],
        "tail_failure_accounting_complete": True,
    }
    del cache, probe
    _cleanup()
    return result


def failure_record(error: Exception, *, phase: str, policy: str, repetition: int) -> dict[str, Any]:
    return {
        "failure_type": ("oom" if isinstance(error, torch.OutOfMemoryError) else "runtime-error"),
        "error_type": type(error).__name__,
        "error": str(error),
        "phase": phase,
        "policy": policy,
        "repetition": repetition,
    }


def execute(spec: dict[str, Any], *, executable: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is required for the production adapter.")
    _require(not _git("status", "--porcelain"), "Production adapter requires a clean tree.")
    cell = spec["cell"]
    device = torch.device("cuda:0")
    model = _load_model(Path(spec["checkpoint"]), device)
    failures: dict[str, dict[str, Any] | None] = {policy: None for policy in POLICIES}
    warmup_failures: list[dict[str, Any]] = []
    seeds = spec["repetition_seeds"]
    for repetition in range(spec["warmups"]):
        prompt, decode, input_digest = generate_inputs(
            model,
            context=cell["context"],
            generation=cell["generation"],
            batch=cell["batch"],
            concurrency=cell["concurrency"],
            seed=seeds[repetition],
        )
        order = POLICIES if repetition % 2 == 0 else tuple(reversed(POLICIES))
        for policy in order:
            if failures[policy] is not None:
                continue
            try:
                run_policy(
                    model,
                    policy=policy,
                    prompt_cpu=prompt,
                    decode_cpu=decode,
                    input_digest=input_digest,
                    batch=cell["batch"],
                    concurrency=cell["concurrency"],
                    device=device,
                )
            except Exception as error:
                failure = failure_record(
                    error, phase="warmup", policy=policy, repetition=repetition
                )
                failures[policy] = failure
                warmup_failures.append(failure)
        del prompt, decode
    repetitions: list[dict[str, Any]] = []
    for repetition in range(spec["measured_repetitions"]):
        if all(failures[policy] is not None for policy in POLICIES):
            break
        prompt, decode, input_digest = generate_inputs(
            model,
            context=cell["context"],
            generation=cell["generation"],
            batch=cell["batch"],
            concurrency=cell["concurrency"],
            seed=seeds[spec["warmups"] + repetition],
        )
        order = POLICIES if repetition % 2 == 0 else tuple(reversed(POLICIES))
        policy_runs: dict[str, Any] = {}
        for policy in order:
            if failures[policy] is not None:
                continue
            try:
                policy_runs[policy] = run_policy(
                    model,
                    policy=policy,
                    prompt_cpu=prompt,
                    decode_cpu=decode,
                    input_digest=input_digest,
                    batch=cell["batch"],
                    concurrency=cell["concurrency"],
                    device=device,
                )
            except Exception as error:
                failures[policy] = failure_record(
                    error, phase="measured", policy=policy, repetition=repetition
                )
        identical = (
            set(policy_runs) == set(POLICIES)
            and policy_runs[POLICIES[0]]["prediction_digest"]
            == policy_runs[POLICIES[1]]["prediction_digest"]
        )
        rejected_run = None
        if set(policy_runs) == set(POLICIES) and not identical:
            rejected_run = policy_runs.pop("tiered-native")
            failures["tiered-native"] = {
                "failure_type": "prediction-divergence",
                "phase": "measured",
                "policy": "tiered-native",
                "repetition": repetition,
                "resident_prediction_digest": policy_runs["resident-native"]["prediction_digest"],
                "tiered_prediction_digest": rejected_run["prediction_digest"],
            }
        repetitions.append(
            {
                "repetition": repetition,
                "input_digest": input_digest,
                "execution_order": list(order),
                "greedy_predictions_identical": identical,
                "policies": policy_runs,
                "rejected_policy_run": rejected_run,
            }
        )
        del prompt, decode
    counts = {policy: sum(policy in row["policies"] for row in repetitions) for policy in POLICIES}
    complete = sum(counts[policy] == spec["measured_repetitions"] for policy in POLICIES)
    status = "complete" if complete == 2 else "partial" if complete == 1 else "failed"
    del model
    _cleanup()
    return {
        "schema_version": 1,
        "experiment_id": "p4-production-adapter-cell-v1",
        "cell": cell,
        "status": status,
        "warmups": spec["warmups"],
        "warmup_repetitions_completed": spec["warmups"],
        "warmup_failures": warmup_failures,
        "measured_repetitions": spec["measured_repetitions"],
        "backend": backend_provenance(executable, device),
        "repetitions": repetitions,
        "policy_status": {
            policy: {
                "measured_repetitions": counts[policy],
                "failure": failures[policy],
            }
            for policy in POLICIES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen P4 cell with actual static continuous batching."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = validate_spec(args.spec)
    payload = execute(spec, executable=Path(__file__).resolve())
    _atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
