from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock

from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM, measure_cache_memory

SCALES = ("s55", "s151")
CONTEXTS = (8_192, 32_768, 131_072)
GENERATIONS = (128, 512, 2_048)
LOAD_PROFILES = (
    ("batch-b1", 1, 1),
    ("batch-b4", 4, 1),
    ("batch-b8", 8, 1),
    ("batch-b16", 16, 1),
    ("interleaved-c8", 1, 8),
    ("interleaved-b4-c8", 4, 8),
    ("interleaved-b8-c8", 8, 8),
    ("interleaved-b16-c8", 16, 8),
    ("interleaved-c32", 1, 32),
    ("interleaved-b4-c32", 4, 32),
    ("interleaved-b8-c32", 8, 32),
    ("interleaved-b16-c32", 16, 32),
)
POLICIES = ("resident-native", "tiered-native")
P3_BENCHMARKS = ("RULER", "SCBench", "LongBench-v2", "LongMemEval", "MRCR")
TERMINAL_STATUSES = ("complete", "partial", "failed")
WARMUPS = 5
MEASURED_REPETITIONS = 30
PREFILL_CHUNK = 256
EXPECTED_CELLS = len(SCALES) * len(CONTEXTS) * len(GENERATIONS) * len(LOAD_PROFILES)
IMPLEMENTATION_PATHS = (
    "nano_deepseek_v4",
    "research/adaptive_v4_memory/manifests/p4-reference-systems-matrix-v1.json",
    "research/adaptive_v4_memory/scripts/adaptive_v4_gpu_lock.py",
    "research/adaptive_v4_memory/scripts/run_p4_systems_matrix.py",
)


@dataclass
class IndexerTimingTrace:
    indexer_time_ns: int = 0
    selection_calls: int = 0

    def record_csa_selection(self, **values: Any) -> None:
        self.indexer_time_ns += int(values["indexer_wall_time_ns"])
        self.selection_calls += 1

    def record_cache_advance(self, *_args: Any, **_kwargs: Any) -> None:
        return


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_digest() -> str:
    tree = subprocess.run(
        ["git", "ls-files", "-s", "--", *IMPLEMENTATION_PATHS],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = {line.split("\t", 1)[1] for line in tree.splitlines() if "\t" in line}
    missing = [
        path
        for path in IMPLEMENTATION_PATHS
        if path not in paths
        and not any(candidate.startswith(path.rstrip("/") + "/") for candidate in paths)
    ]
    if missing:
        raise RuntimeError(f"Untracked P4 implementation paths: {missing}")
    return hashlib.sha256(tree.encode()).hexdigest()


def _dirty() -> bool:
    return bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def require_p3_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    audit = payload.get("audit", {})
    benchmarks = payload.get("benchmarks", {})
    if (
        payload.get("experiment_id") != "p3-natural-language-suite-audit-v1"
        or audit.get("all_required_artifacts_verified") is not True
        or audit.get("all_required_baseline_cells_terminal") is not True
        or audit.get("all_failure_accounting_complete") is not True
        or audit.get("safety_stress_terminal") is not True
        or audit.get("natural_safety_terminal") is not True
        or audit.get("benchmarks_terminal") != len(P3_BENCHMARKS)
        or audit.get("minimum_protocol_examples_accounted_per_arm") != 45_289
        or set(benchmarks) != set(P3_BENCHMARKS)
        or payload.get("supplemental_safety", {}).get("terminal") is not True
        or payload.get("supplemental_safety", {}).get("protected_prefix_physical_budget_verified")
        is not True
        or payload.get("supplemental_natural_safety", {}).get("terminal") is not True
        or payload.get("supplemental_natural_safety", {}).get("longsafety_official_judge_status")
        != "blocked"
        or payload.get("supplemental_natural_safety", {}).get(
            "comparative_long_context_safety_claim_available"
        )
        is not False
        or any(
            benchmarks[name].get("terminal") is not True
            or benchmarks[name].get("native_and_fixed_terminal") is not True
            for name in P3_BENCHMARKS
        )
    ):
        raise RuntimeError(
            "P4 is deferred until the complete digest-bound five-benchmark and safety P3 audit."
        )
    return payload


def frozen_cells() -> tuple[tuple[str, int, int, str, int, int], ...]:
    return tuple(
        (scale, context, generation, name, batch, active_requests)
        for scale, context, generation, (name, batch, active_requests) in product(
            SCALES, CONTEXTS, GENERATIONS, LOAD_PROFILES
        )
    )


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("A latency distribution must not be empty.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "observations": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "maximum_ms": max(values),
    }


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
    active_requests: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], str]:
    generator = torch.Generator().manual_seed(seed)
    prompts = [
        torch.randint(0, model.config.vocab_size, (batch, context), generator=generator)
        for _ in range(active_requests)
    ]
    decode = [
        torch.randint(0, model.config.vocab_size, (batch, generation), generator=generator)
        for _ in range(active_requests)
    ]
    return prompts, decode, _tensor_digest([*prompts, *decode])


def _cache_totals(caches: list[Any]) -> dict[str, int]:
    accounting = [measure_cache_memory(cache) for cache in caches]
    stores = [item for cache in caches for item in cache.tiered_memory_stats()]
    return {
        "logical_cache_bytes": sum(item.logical_cache_bytes for item in accounting),
        "hot_resident_bytes": sum(item.hot_resident_bytes for item in accounting),
        "cold_resident_bytes": sum(item.cold_resident_bytes for item in accounting),
        "pinned_host_bytes": sum(item.host_bytes for item in stores),
        "tier_hot_bytes": sum(item.hot_bytes for item in stores),
        "h2d_bytes": sum(item.h2d_bytes for item in stores),
        "d2h_bytes": sum(item.d2h_bytes for item in stores),
        "h2d_count": sum(item.h2d_count for item in stores),
        "d2h_count": sum(item.d2h_count for item in stores),
        "useful_h2d_bytes": sum(item.useful_h2d_bytes for item in stores),
        "late_misses": sum(item.late_misses for item in stores),
        "prefetches": sum(item.prefetches for item in stores),
        "evictions": sum(item.evictions for item in stores),
    }


@torch.inference_mode()
def run_policy(
    model: DeepSeekV4ForCausalLM,
    *,
    policy: str,
    prompts: list[torch.Tensor],
    decode_tokens: list[torch.Tensor],
    input_digest: str,
    device: torch.device,
) -> dict[str, Any]:
    if policy not in POLICIES:
        raise ValueError(f"Unknown P4 policy: {policy}")
    _cleanup()
    baseline_allocated = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    caches: list[Any] = []
    request_prefill_ms: list[float] = []
    cell_started = time.perf_counter_ns()
    for prompt_cpu in prompts:
        request_started = time.perf_counter_ns()
        cache: Any = None
        for start in range(0, prompt_cpu.shape[1], PREFILL_CHUNK):
            chunk = prompt_cpu[:, start : start + PREFILL_CHUNK].to(device)
            output = model(chunk, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if cache is None:
                raise RuntimeError("P4 prefill did not return a cache.")
            if policy == "tiered-native" and start == 0:
                cache.enable_csa_tiering(model.config.index_topk * prompt_cpu.shape[0])
            del output, chunk
        torch.cuda.synchronize()
        request_prefill_ms.append((time.perf_counter_ns() - request_started) / 1_000_000.0)
        caches.append(cache)
    allocated_after_prefill = torch.cuda.memory_allocated(device)
    reserved_after_prefill = torch.cuda.memory_reserved(device)
    prefill_finished = time.perf_counter_ns()
    first_token_completion_ms: list[float] = []
    step_latencies_ms: list[float] = []
    prediction_digest = hashlib.sha256()
    decode_started = time.perf_counter_ns()
    generation = decode_tokens[0].shape[1]
    for token_index in range(generation):
        for request_index, cache in enumerate(caches):
            token = decode_tokens[request_index][:, token_index : token_index + 1].to(device)
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            output = model(token, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            step_latencies_ms.append(elapsed_ms)
            greedy = output.logits[:, -1].argmax(dim=-1).to(device="cpu")
            prediction_digest.update(greedy.numpy().tobytes())
            if token_index == 0:
                first_token_completion_ms.append(
                    (time.perf_counter_ns() - cell_started) / 1_000_000.0
                )
            del output, token, greedy
    torch.cuda.synchronize()
    decode_elapsed_s = (time.perf_counter_ns() - decode_started) / 1_000_000_000.0
    end_to_end_ms = (time.perf_counter_ns() - cell_started) / 1_000_000.0
    totals = _cache_totals(caches)
    trace = IndexerTimingTrace()
    caches[0].memory_trace = trace
    probe = torch.zeros((prompts[0].shape[0], 1), dtype=torch.long, device=device)
    model(probe, past_key_values=caches[0], use_cache=True)
    torch.cuda.synchronize()
    caches[0].memory_trace = None
    generated_tokens = sum(tokens.shape[0] * tokens.shape[1] for tokens in decode_tokens)
    result = {
        "policy": policy,
        "input_digest": input_digest,
        "prediction_digest": prediction_digest.hexdigest(),
        "requests": len(caches),
        "batch": prompts[0].shape[0],
        "context_tokens": prompts[0].shape[1],
        "generation_tokens": generation,
        "load_execution": {
            "model": "serial-round-robin-interleave",
            "active_requests": len(caches),
            "actual_concurrent_serving": False,
        },
        "request_prefill_ms": latency_summary(request_prefill_ms),
        "aggregate_prefill_ms": (prefill_finished - cell_started) / 1_000_000.0,
        "ttft_ms": latency_summary(first_token_completion_ms),
        "decode_step_ms": latency_summary(step_latencies_ms),
        "end_to_end_ms": end_to_end_ms,
        "generated_token_throughput_per_second": generated_tokens / decode_elapsed_s,
        "cuda": {
            "cache_allocated_delta_bytes": allocated_after_prefill - baseline_allocated,
            "allocated_after_prefill_bytes": allocated_after_prefill,
            "reserved_after_prefill_bytes": reserved_after_prefill,
            "fragmentation_after_prefill_bytes": (reserved_after_prefill - allocated_after_prefill),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "cache": totals,
        "transfer": {
            "useful_h2d_ratio": totals["useful_h2d_bytes"] / max(totals["h2d_bytes"], 1),
        },
        "untimed_indexer_probe": asdict(trace),
        "controller_time_ns": 0,
    }
    del caches, probe
    _cleanup()
    return result


def cell_path(root: Path, cell: tuple[str, int, int, str, int, int]) -> Path:
    scale, context, generation, profile, _batch, _active_requests = cell
    return root / scale / f"context-{context}" / f"generation-{generation}" / profile


def _artifact_valid(
    path: Path,
    *,
    cell: tuple[str, int, int, str, int, int],
    digest: str,
    manifest_digest: str,
    p3_digest: str,
) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    identity_ok = (
        payload.get("schema_version") == 1
        and payload.get("experiment_id") == "p4-reference-systems-cell-v1"
        and tuple(
            payload.get("cell", {}).get(name)
            for name in ("scale", "context", "generation", "profile", "batch", "active_requests")
        )
        == cell
        and payload.get("source", {}).get("implementation_digest") == digest
        and payload.get("manifest", {}).get("sha256") == manifest_digest
        and payload.get("p3_audit", {}).get("sha256") == p3_digest
        and payload.get("status") in TERMINAL_STATUSES
    )
    if not identity_ok:
        return False
    repetitions = payload.get("repetitions", [])
    warmup_attempted = payload.get("warmup_repetitions_attempted")
    warmup_paired = payload.get("warmup_paired_repetitions_completed")
    warmup_policy_runs = payload.get("warmup_policy_runs_completed")
    warmup_failures = payload.get("warmup_failures")
    if not (
        payload.get("warmups") == WARMUPS
        and payload.get("warmup_accounting_available") is True
        and isinstance(warmup_attempted, int)
        and 0 <= warmup_attempted <= WARMUPS
        and isinstance(warmup_paired, int)
        and 0 <= warmup_paired <= warmup_attempted
        and isinstance(warmup_policy_runs, dict)
        and set(warmup_policy_runs) == set(POLICIES)
        and all(
            isinstance(warmup_policy_runs[policy], int)
            and 0 <= warmup_policy_runs[policy] <= warmup_attempted
            for policy in POLICIES
        )
        and warmup_paired == min(warmup_policy_runs.values())
        and isinstance(warmup_failures, list)
        and payload.get("measured_repetitions") == MEASURED_REPETITIONS
        and isinstance(repetitions, list)
        and len(repetitions) <= MEASURED_REPETITIONS
        and all(_valid_repetition(row, index) for index, row in enumerate(repetitions))
    ):
        return False
    policy_status = payload.get("policy_status", {})
    if set(policy_status) != set(POLICIES):
        return False
    counts = {
        policy: sum(policy in row.get("policies", {}) for row in repetitions) for policy in POLICIES
    }
    if any(
        policy_status[policy].get("measured_repetitions") != counts[policy] for policy in POLICIES
    ):
        return False
    for policy in POLICIES:
        failure = policy_status[policy].get("failure")
        if warmup_policy_runs[policy] < WARMUPS:
            if not (
                isinstance(failure, dict)
                and failure.get("phase") == "warmup"
                and failure in warmup_failures
            ):
                return False
        elif isinstance(failure, dict) and failure.get("phase") == "warmup":
            return False
    if len(warmup_failures) != sum(
        isinstance(policy_status[policy].get("failure"), dict)
        and policy_status[policy]["failure"].get("phase") == "warmup"
        for policy in POLICIES
    ):
        return False
    complete = {policy for policy in POLICIES if counts[policy] == MEASURED_REPETITIONS}
    expected_status = (
        "complete" if len(complete) == len(POLICIES) else "partial" if complete else "failed"
    )
    return payload["status"] == expected_status and all(
        counts[policy] == MEASURED_REPETITIONS
        or isinstance(policy_status[policy].get("failure"), dict)
        for policy in POLICIES
    )


def _valid_repetition(row: dict[str, Any], index: int) -> bool:
    policies = row.get("policies", {})
    input_digest = row.get("input_digest")
    expected_order = POLICIES if index % 2 == 0 else tuple(reversed(POLICIES))
    if not (
        row.get("repetition") == index
        and isinstance(input_digest, str)
        and len(input_digest) == 64
        and set(input_digest) <= set("0123456789abcdef")
        and tuple(row.get("execution_order", ())) == expected_order
        and isinstance(policies, dict)
        and set(policies).issubset(POLICIES)
        and all(input_digest == policy_run.get("input_digest") for policy_run in policies.values())
    ):
        return False
    if set(policies) != set(POLICIES):
        return True
    identical = (
        policies[POLICIES[0]].get("prediction_digest")
        == policies[POLICIES[1]].get("prediction_digest")
    )
    return row.get("greedy_predictions_identical") is identical


def _run_row(
    cell: tuple[str, int, int, str, int, int], artifact: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "scale": cell[0],
        "context": cell[1],
        "generation": cell[2],
        "profile": cell[3],
        "batch": cell[4],
        "active_requests": cell[5],
        "status": payload["status"],
        "artifact": {"path": str(artifact), "sha256": sha256(artifact)},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_matrix(
    path: Path,
    *,
    runs: list[dict[str, Any]],
    implementation: str,
    manifest: Path,
    p3_audit: Path,
) -> None:
    runs.sort(
        key=lambda row: tuple(row[name] for name in ("scale", "context", "generation", "profile"))
    )
    _write_json(
        path,
        {
            "schema_version": 1,
            "experiment_id": "p4-reference-systems-matrix-progress-v1",
            "source_commit": _head(),
            "implementation_digest": implementation,
            "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
            "p3_audit": {"path": str(p3_audit), "sha256": sha256(p3_audit)},
            "expected_cells": EXPECTED_CELLS,
            "terminal_cells": len(runs),
            "complete_cells": sum(row["status"] == "complete" for row in runs),
            "partial_cells": sum(row["status"] == "partial" for row in runs),
            "failed_cells": sum(row["status"] == "failed" for row in runs),
            "runs": runs,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen P4 reference systems matrix.")
    parser.add_argument("--scale", action="append", choices=SCALES)
    parser.add_argument("--context", type=int, action="append", choices=CONTEXTS)
    parser.add_argument("--generation", type=int, action="append", choices=GENERATIONS)
    parser.add_argument(
        "--profile", action="append", choices=tuple(row[0] for row in LOAD_PROFILES)
    )
    parser.add_argument("--max-new-cells", type=int)
    parser.add_argument("--cell-timeout-seconds", type=float, default=21_600.0)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/training"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p4-reference-systems-matrix-v1.json"),
    )
    parser.add_argument(
        "--p3-audit",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-suite.summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/reference-systems"),
    )
    parser.add_argument(
        "--matrix-progress",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p4/reference-systems-matrix.json"),
    )
    args = parser.parse_args()
    if args.max_new_cells is not None and args.max_new_cells <= 0:
        raise ValueError("max-new-cells must be positive.")
    if args.cell_timeout_seconds <= 0:
        raise ValueError("cell-timeout-seconds must be positive.")
    if _dirty():
        raise RuntimeError("P4 systems execution requires a clean source tree.")
    require_p3_audit(args.p3_audit)
    manifest = json.loads(args.manifest.read_text())
    if (
        manifest.get("experiment_id") != "p4-reference-systems-matrix-v1"
        or manifest.get("primary_paired_cells") != EXPECTED_CELLS
    ):
        raise RuntimeError("The frozen P4 systems manifest is required.")
    if not torch.cuda.is_available():
        raise RuntimeError("P4 systems execution requires CUDA.")
    lock = acquire_gpu_lock("p4-reference-systems-matrix")
    implementation = implementation_digest()
    manifest_digest = sha256(args.manifest)
    p3_digest = sha256(args.p3_audit)
    selected = [
        cell
        for cell in frozen_cells()
        if (not args.scale or cell[0] in args.scale)
        and (not args.context or cell[1] in args.context)
        and (not args.generation or cell[2] in args.generation)
        and (not args.profile or cell[3] in args.profile)
    ]
    runs_by_cell: dict[tuple[str, int, int, str, int, int], dict[str, Any]] = {}
    for existing_cell in frozen_cells():
        existing_artifact = cell_path(args.output_root, existing_cell) / "cell.json"
        if _artifact_valid(
            existing_artifact,
            cell=existing_cell,
            digest=implementation,
            manifest_digest=manifest_digest,
            p3_digest=p3_digest,
        ):
            existing_payload = json.loads(existing_artifact.read_text())
            runs_by_cell[existing_cell] = _run_row(
                existing_cell, existing_artifact, existing_payload
            )
    new_cells = 0
    models: dict[str, DeepSeekV4ForCausalLM] = {}
    device = torch.device("cuda")
    for cell in selected:
        root = cell_path(args.output_root, cell)
        artifact = root / "cell.json"
        if _artifact_valid(
            artifact,
            cell=cell,
            digest=implementation,
            manifest_digest=manifest_digest,
            p3_digest=p3_digest,
        ):
            payload = json.loads(artifact.read_text())
        else:
            if args.max_new_cells is not None and new_cells >= args.max_new_cells:
                break
            scale, context, generation, profile, batch, active_requests = cell
            if scale not in models:
                checkpoint = args.training_root / scale / "seed-6071401" / f"{scale}-step-1000.pt"
                models[scale] = _load_model(checkpoint, device)
            model = models[scale]
            started = time.monotonic()
            repetitions: list[dict[str, Any]] = []
            policy_failures: dict[str, dict[str, Any]] = {}
            warmup_repetitions_attempted = 0
            warmup_paired_repetitions_completed = 0
            warmup_policy_runs_completed = {policy: 0 for policy in POLICIES}
            warmup_failures: list[dict[str, Any]] = []
            try:
                for repetition in range(WARMUPS + MEASURED_REPETITIONS):
                    if time.monotonic() - started > args.cell_timeout_seconds:
                        for policy in POLICIES:
                            if policy not in policy_failures:
                                failure = {
                                    "failure_type": "timeout",
                                    "error_type": "TimeoutError",
                                    "error": "P4 paired cell exceeded its frozen wall-time limit.",
                                    "phase": ("warmup" if repetition < WARMUPS else "measured"),
                                    "repetition": repetition,
                                }
                                policy_failures[policy] = failure
                                if repetition < WARMUPS:
                                    warmup_failures.append(failure)
                        break
                    prompts, decode, input_digest = generate_inputs(
                        model,
                        context=context,
                        generation=generation,
                        batch=batch,
                        active_requests=active_requests,
                        seed=9_071_400 + repetition,
                    )
                    if repetition < WARMUPS:
                        warmup_repetitions_attempted += 1
                    order = POLICIES if repetition % 2 == 0 else tuple(reversed(POLICIES))
                    policy_runs: dict[str, dict[str, Any]] = {}
                    failures_this_repetition: dict[str, dict[str, Any]] = {}
                    for policy in order:
                        if policy in policy_failures:
                            continue
                        try:
                            policy_runs[policy] = run_policy(
                                model,
                                policy=policy,
                                prompts=prompts,
                                decode_tokens=decode,
                                input_digest=input_digest,
                                device=device,
                            )
                            if repetition < WARMUPS:
                                warmup_policy_runs_completed[policy] += 1
                        except Exception as error:
                            failure = {
                                "failure_type": (
                                    "oom"
                                    if isinstance(error, torch.cuda.OutOfMemoryError)
                                    else "error"
                                ),
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "phase": ("warmup" if repetition < WARMUPS else "measured"),
                                "repetition": repetition,
                            }
                            policy_failures[policy] = failure
                            failures_this_repetition[policy] = failure
                            if repetition < WARMUPS:
                                warmup_failures.append(failure)
                            _cleanup()
                    if repetition < WARMUPS and set(policy_runs) == set(POLICIES):
                        warmup_paired_repetitions_completed += 1
                    if repetition >= WARMUPS:
                        paired = set(policy_runs) == set(POLICIES)
                        repetitions.append(
                            {
                                "repetition": repetition - WARMUPS,
                                "execution_order": order,
                                "input_digest": input_digest,
                                "greedy_predictions_identical": (
                                    policy_runs[POLICIES[0]]["prediction_digest"]
                                    == policy_runs[POLICIES[1]]["prediction_digest"]
                                    if paired
                                    else None
                                ),
                                "policies": policy_runs,
                                "policy_failures": failures_this_repetition,
                            }
                        )
                    del prompts, decode, policy_runs
                    if len(policy_failures) == len(POLICIES):
                        break
                measured_counts = {
                    policy: sum(policy in row["policies"] for row in repetitions)
                    for policy in POLICIES
                }
                policy_status = {
                    policy: {
                        "status": (
                            "complete"
                            if measured_counts[policy] == MEASURED_REPETITIONS
                            else "failed"
                        ),
                        "measured_repetitions": measured_counts[policy],
                        "failure": policy_failures.get(policy),
                    }
                    for policy in POLICIES
                }
                completed_policies = sum(
                    row["status"] == "complete" for row in policy_status.values()
                )
                status = (
                    "complete"
                    if completed_policies == len(POLICIES)
                    else "partial"
                    if completed_policies
                    else "failed"
                )
                payload = {
                    "schema_version": 1,
                    "experiment_id": "p4-reference-systems-cell-v1",
                    "status": status,
                    "cell": {
                        "scale": scale,
                        "context": context,
                        "generation": generation,
                        "profile": profile,
                        "batch": batch,
                        "active_requests": active_requests,
                    },
                    "warmups": WARMUPS,
                    "warmup_accounting_available": True,
                    "warmup_repetitions_attempted": warmup_repetitions_attempted,
                    "warmup_paired_repetitions_completed": (
                        warmup_paired_repetitions_completed
                    ),
                    "warmup_policy_runs_completed": warmup_policy_runs_completed,
                    "warmup_failures": warmup_failures,
                    "measured_repetitions": MEASURED_REPETITIONS,
                    "repetitions": repetitions,
                    "policy_status": policy_status,
                    "all_available_paired_predictions_identical": all(
                        row["greedy_predictions_identical"] is not False for row in repetitions
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                }
            except Exception as error:
                failure_type = (
                    "oom"
                    if isinstance(error, torch.cuda.OutOfMemoryError)
                    else "timeout"
                    if isinstance(error, TimeoutError)
                    else "error"
                )
                orchestration_phase = (
                    "warmup"
                    if any(
                        warmup_policy_runs_completed[policy] < WARMUPS
                        for policy in POLICIES
                    )
                    else "measured"
                )
                terminal_failures = {
                    policy: policy_failures.get(policy)
                    or {
                        "failure_type": failure_type,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "phase": orchestration_phase,
                    }
                    for policy in POLICIES
                }
                warmup_failures = [
                    failure
                    for failure in terminal_failures.values()
                    if failure.get("phase") == "warmup"
                ]
                payload = {
                    "schema_version": 1,
                    "experiment_id": "p4-reference-systems-cell-v1",
                    "status": "failed",
                    "failure_type": failure_type,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "cell": {
                        "scale": scale,
                        "context": context,
                        "generation": generation,
                        "profile": profile,
                        "batch": batch,
                        "active_requests": active_requests,
                    },
                    "warmups": WARMUPS,
                    "warmup_accounting_available": True,
                    "warmup_repetitions_attempted": warmup_repetitions_attempted,
                    "warmup_paired_repetitions_completed": (
                        warmup_paired_repetitions_completed
                    ),
                    "warmup_policy_runs_completed": warmup_policy_runs_completed,
                    "warmup_failures": warmup_failures,
                    "measured_repetitions": MEASURED_REPETITIONS,
                    "repetitions": repetitions,
                    "policy_status": {
                        policy: {
                            "status": "failed",
                            "measured_repetitions": sum(
                                policy in row.get("policies", {}) for row in repetitions
                            ),
                            "failure": terminal_failures[policy],
                        }
                        for policy in POLICIES
                    },
                    "completed_measured_repetitions": len(repetitions),
                    "elapsed_seconds": time.monotonic() - started,
                }
                _cleanup()
            payload.update(
                {
                    "source": {
                        "commit": _head(),
                        "dirty": False,
                        "implementation_digest": implementation,
                    },
                    "manifest": {"path": str(args.manifest), "sha256": manifest_digest},
                    "p3_audit": {"path": str(args.p3_audit), "sha256": p3_digest},
                    "environment": {
                        "python": platform.python_version(),
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "device": torch.cuda.get_device_name(device),
                    },
                    "command": [sys.executable, *sys.argv],
                }
            )
            _write_json(artifact, payload)
            new_cells += 1
        runs_by_cell[cell] = _run_row(cell, artifact, payload)
        _write_matrix(
            args.matrix_progress,
            runs=list(runs_by_cell.values()),
            implementation=implementation,
            manifest=args.manifest,
            p3_audit=args.p3_audit,
        )
        print(json.dumps({"cell": payload["cell"], "status": payload["status"]}), flush=True)
    lock.close()


if __name__ == "__main__":
    main()
