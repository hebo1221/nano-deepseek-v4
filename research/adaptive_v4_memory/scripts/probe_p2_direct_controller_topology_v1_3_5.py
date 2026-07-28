from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import p2_direct_attestation as attestation
import p2_direct_controller_contract_v1_3_5 as contract
import p2_direct_controller_reuse_admission_v1_3 as legacy
from adaptive_v4_gpu_lock import (
    GPULockLease,
    acquire_device_guard,
    acquire_gpu_lock,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

# This v3 system-only probe replaces the unexecuted /tmp v2 draft.  It keeps the
# frozen 1/2/3/4 order and selector, but removes held-out examples entirely to
# avoid a circular dependency on the not-yet-published v1.3.5 manifest.  Each
# child loads the exact admitted s151 checkpoint and runs the same deterministic
# synthetic autoregressive trace.  Full17 semantics remain guarded by the fresh
# canonical one-cell invariant gate after the selected count is frozen.
PROBE_SCHEMA_VERSION = 3
PROBE_EXPERIMENT_ID = "p2-direct-controller-same-gpu-system-topology-probe-v3"
PROBE_ATTESTATION_PURPOSE = "p2-direct-controller-same-gpu-system-topology-probe-v3"
PROBE_OUTPUT_ROOT = Path(
    "artifacts/adaptive_v4_memory/paper_grade/p2_post_rank_direct/topology-probe-v3"
)
PROBE_RECEIPT_PATH = PROBE_OUTPUT_ROOT / "topology-probe.summary.json"
CANDIDATE_WORKER_COUNTS = (1, 2, 3, 4)
SELECTED_SCALE = "s151"
SELECTED_TRAINING_SEED = 6_071_406
PHASE_A_PROMPT_TOKENS = 256
PHASE_A_DECODE_TOKENS = 64
PHASE_B_PROMPT_TOKENS = 1024
PHASE_B_DECODE_TOKENS = 192
READY_TIMEOUT_SECONDS = 900
CHILD_TIMEOUT_SECONDS = 1800
MINIMUM_AVAILABLE_RAM_BYTES = 16 << 30
MAXIMUM_SWAP_INCREASE_BYTES = 256 << 20
MINIMUM_USEFUL_ACCELERATION = 1.5
NEAR_BEST_FRACTION = 0.95

_CHILD_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "worker_index",
        "worker_count",
        "scale",
        "training_seed",
        "prompt_tokens",
        "decode_tokens",
        "synthetic_trace_sha256",
        "device_name",
        "device_capability",
        "cuda_total_memory_bytes",
        "model_load_seconds",
        "service_seconds",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "quality_values_accessed",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    legacy._write_exclusive_durable(path, legacy.canonical_pretty_json(payload))


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, separator, raw = line.partition(":")
        if separator != ":":
            continue
        fields = raw.strip().split()
        if fields and fields[0].isdigit():
            multiplier = 1024 if len(fields) > 1 and fields[1] == "kB" else 1
            values[name] = int(fields[0]) * multiplier
    return {
        "available_ram_bytes": values.get("MemAvailable", 0),
        "swap_used_bytes": max(
            0,
            values.get("SwapTotal", 0) - values.get("SwapFree", 0),
        ),
    }


def _process_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


def _gpu_sample() -> dict[str, float | str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {"status": "unsupported"}
    fields = [field.strip() for field in completed.stdout.splitlines()[0].split(",")]
    try:
        return {
            "status": "supported",
            "gpu_utilization_percent": float(fields[0]),
            "memory_utilization_percent": float(fields[1]),
            "power_watts": float(fields[2]),
        }
    except (IndexError, ValueError):
        return {"status": "unsupported"}


def _sample_resources(
    processes: Sequence[subprocess.Popen[str]],
    stop: threading.Event,
    output: dict[str, Any],
) -> None:
    minimum_available = math.inf
    maximum_swap = 0
    maximum_rss = 0
    maximum_gpu = 0.0
    maximum_memory = 0.0
    maximum_power = 0.0
    gpu_supported = False
    samples = 0
    while not stop.wait(0.25):
        memory = _meminfo()
        minimum_available = min(minimum_available, memory["available_ram_bytes"])
        maximum_swap = max(maximum_swap, memory["swap_used_bytes"])
        maximum_rss = max(
            maximum_rss,
            sum(_process_rss_bytes(process.pid) for process in processes),
        )
        if samples % 4 == 0:
            gpu = _gpu_sample()
            if gpu.get("status") == "supported":
                gpu_supported = True
                maximum_gpu = max(
                    maximum_gpu,
                    cast(float, gpu["gpu_utilization_percent"]),
                )
                maximum_memory = max(
                    maximum_memory,
                    cast(float, gpu["memory_utilization_percent"]),
                )
                maximum_power = max(maximum_power, cast(float, gpu["power_watts"]))
        samples += 1
    if math.isinf(minimum_available):
        memory = _meminfo()
        minimum_available = memory["available_ram_bytes"]
        maximum_swap = memory["swap_used_bytes"]
    output.update(
        {
            "sample_period_milliseconds": 250,
            "sample_count": samples,
            "minimum_available_ram_bytes": int(minimum_available),
            "maximum_swap_used_bytes": maximum_swap,
            "maximum_combined_child_rss_bytes": maximum_rss,
            "gpu_metrics": (
                {
                    "status": "supported",
                    "maximum_gpu_utilization_percent": maximum_gpu,
                    "maximum_memory_utilization_percent": maximum_memory,
                    "maximum_power_watts": maximum_power,
                }
                if gpu_supported
                else {"status": "unsupported"}
            ),
        }
    )


def _device_identity() -> dict[str, str]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid = completed.stdout.splitlines()[0].strip()
    _require(uuid.startswith("GPU-"), "Selected GPU UUID is unavailable.")
    return {"identity_type": "uuid", "identity": uuid}


def _verify_inherited_lease(prefix: str) -> None:
    descriptor = int(os.environ[f"{prefix}_FD"])
    expected_device = int(os.environ[f"{prefix}_DEVICE"])
    expected_inode = int(os.environ[f"{prefix}_INODE"])
    path = Path(os.environ[f"{prefix}_PATH"])
    opened = os.fstat(descriptor)
    current = os.stat(path, follow_symlinks=False)
    _require(
        (opened.st_dev, opened.st_ino)
        == (current.st_dev, current.st_ino)
        == (expected_device, expected_inode),
        f"Inherited {prefix.lower()} lease identity drifted.",
    )


def _publish_ready(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, b"ready\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _child_main(args: argparse.Namespace) -> int:
    _verify_inherited_lease("PROBE_SCHEDULER")
    _verify_inherited_lease("PROBE_DEVICE_GUARD")
    import torch
    from calibrate_p2_direct_soft_lag import _load_checkpoint_model

    from nano_deepseek_v4 import DeepSeekV4Cache

    load_started = time.perf_counter()
    with args.checkpoint.open("rb") as stream:
        raw = torch.load(stream, map_location="cpu", weights_only=True)
    _require(isinstance(raw, Mapping), "Probe checkpoint payload is invalid.")
    device = torch.device("cuda:0")
    model = _load_checkpoint_model(
        cast(Mapping[str, Any], raw),
        scale=SELECTED_SCALE,
        training_seed=SELECTED_TRAINING_SEED,
        device=device,
        dtype=torch.bfloat16,
    )
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    _publish_ready(args.ready_path)
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while not args.start_path.exists():
        _require(time.monotonic() < deadline, "Probe child start barrier timed out.")
        time.sleep(0.01)

    vocab_size = int(model.config.vocab_size)
    prompt = (
        torch.arange(args.prompt_tokens, dtype=torch.long, device=device)
        .remainder(max(2, vocab_size - 1))
        .add(1)
        .unsqueeze(0)
    )
    trace = hashlib.sha256()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    service_started = time.perf_counter()
    with torch.inference_mode():
        cache = DeepSeekV4Cache(model.config)
        output = model(prompt, past_key_values=cache, use_cache=True)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        for _ in range(args.decode_tokens):
            trace.update(int(token.item()).to_bytes(8, "big", signed=False))
            output = model(token, past_key_values=cache, use_cache=True)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    service_seconds = time.perf_counter() - service_started
    properties = torch.cuda.get_device_properties(device)
    result = {
        "schema_version": 1,
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "scale": SELECTED_SCALE,
        "training_seed": SELECTED_TRAINING_SEED,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "synthetic_trace_sha256": trace.hexdigest(),
        "device_name": properties.name,
        "device_capability": [properties.major, properties.minor],
        "cuda_total_memory_bytes": properties.total_memory,
        "model_load_seconds": load_seconds,
        "service_seconds": service_seconds,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "quality_values_accessed": False,
    }
    _require(set(result) == _CHILD_RESULT_FIELDS, "Probe child result schema drifted.")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _wait_ready(
    processes: Sequence[subprocess.Popen[str]],
    ready_paths: Sequence[Path],
) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    while True:
        if all(path.is_file() and path.read_bytes() == b"ready\n" for path in ready_paths):
            return
        failed = [
            (index, process.returncode)
            for index, process in enumerate(processes)
            if process.poll() is not None
        ]
        _require(not failed, f"Probe child exited before ready barrier: {failed}")
        _require(time.monotonic() < deadline, "Probe ready barrier timed out.")
        time.sleep(0.05)


def _child_command(
    *,
    checkpoint: Path,
    worker_index: int,
    worker_count: int,
    prompt_tokens: int,
    decode_tokens: int,
    ready_path: Path,
    start_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--checkpoint",
        str(checkpoint),
        "--worker-index",
        str(worker_index),
        "--worker-count",
        str(worker_count),
        "--prompt-tokens",
        str(prompt_tokens),
        "--decode-tokens",
        str(decode_tokens),
        "--ready-path",
        str(ready_path),
        "--start-path",
        str(start_path),
    ]


def _run_candidate(
    *,
    phase: str,
    worker_count: int,
    prompt_tokens: int,
    decode_tokens: int,
    checkpoint: Path,
    output_root: Path,
    scheduler_lease: GPULockLease,
    device_guard_lease: GPULockLease,
    reference_digest: str | None,
) -> dict[str, Any]:
    candidate_root = output_root / phase / f"worker-count-{worker_count}"
    _require(
        not os.path.lexists(candidate_root),
        "A probe candidate output already exists; reruns are forbidden.",
    )
    candidate_root.mkdir(parents=True, mode=0o700)
    os.chmod(candidate_root, 0o700)
    start_path = candidate_root / "start.barrier"
    ready_paths = [candidate_root / f"worker-{index}.ready" for index in range(worker_count)]
    environment = os.environ.copy()
    environment.update(
        {
            "PROBE_SCHEDULER_FD": str(scheduler_lease.fileno()),
            "PROBE_SCHEDULER_DEVICE": str(scheduler_lease.device),
            "PROBE_SCHEDULER_INODE": str(scheduler_lease.inode),
            "PROBE_SCHEDULER_PATH": str(scheduler_lease.path),
            "PROBE_DEVICE_GUARD_FD": str(device_guard_lease.fileno()),
            "PROBE_DEVICE_GUARD_DEVICE": str(device_guard_lease.device),
            "PROBE_DEVICE_GUARD_INODE": str(device_guard_lease.inode),
            "PROBE_DEVICE_GUARD_PATH": str(device_guard_lease.path),
        }
    )
    processes = [
        subprocess.Popen(
            _child_command(
                checkpoint=checkpoint,
                worker_index=index,
                worker_count=worker_count,
                prompt_tokens=prompt_tokens,
                decode_tokens=decode_tokens,
                ready_path=ready_paths[index],
                start_path=start_path,
            ),
            cwd=REPOSITORY_ROOT,
            env=environment,
            pass_fds=tuple(sorted({scheduler_lease.fileno(), device_guard_lease.fileno()})),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index in range(worker_count)
    ]
    resource_output: dict[str, Any] = {}
    stop_sampling = threading.Event()
    sampler = threading.Thread(
        target=_sample_resources,
        args=(processes, stop_sampling, resource_output),
        name=f"topology-probe-sampler-{phase}-{worker_count}",
    )
    sampler.start()
    before = _meminfo()
    completed: list[tuple[str, str]] | None = None
    try:
        _wait_ready(processes, ready_paths)
        barrier_descriptor = os.open(
            start_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        os.fsync(barrier_descriptor)
        os.close(barrier_descriptor)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            completed = list(
                pool.map(
                    lambda process: process.communicate(timeout=CHILD_TIMEOUT_SECONDS),
                    processes,
                )
            )
        makespan = time.perf_counter() - started
    finally:
        stop_sampling.set()
        sampler.join(timeout=10)
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
    _require(completed is not None, "Probe candidate ended before child collection.")
    after = _meminfo()
    results: list[dict[str, Any]] = []
    for index, (process, streams) in enumerate(zip(processes, completed, strict=True)):
        stdout, stderr = streams
        if stderr:
            stderr_path = candidate_root / f"worker-{index}.stderr.txt"
            descriptor = os.open(
                stderr_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
            os.write(descriptor, stderr.encode("utf-8", errors="replace"))
            os.fsync(descriptor)
            os.close(descriptor)
        _require(process.returncode == 0, f"Probe child {index} failed.")
        lines = [line for line in stdout.splitlines() if line]
        _require(len(lines) == 1, "Probe child emitted a non-canonical result stream.")
        payload = json.loads(lines[0])
        _require(
            isinstance(payload, dict)
            and set(payload) == _CHILD_RESULT_FIELDS
            and payload.get("worker_index") == index
            and payload.get("worker_count") == worker_count
            and payload.get("quality_values_accessed") is False,
            "Probe child result schema or identity drifted.",
        )
        checked = cast(dict[str, Any], payload)
        _write_json_exclusive(candidate_root / f"worker-{index}.json", checked)
        results.append(checked)
    digests = {cast(str, item["synthetic_trace_sha256"]) for item in results}
    equivalent = len(digests) == 1 and (reference_digest is None or digests == {reference_digest})
    service_times = [float(item["service_seconds"]) for item in results]
    median_service = statistics.median(service_times)
    capacity = worker_count / median_service
    swap_increase = max(
        0,
        resource_output["maximum_swap_used_bytes"] - before["swap_used_bytes"],
        after["swap_used_bytes"] - before["swap_used_bytes"],
    )
    hard_admissible = (
        equivalent
        and resource_output["minimum_available_ram_bytes"] >= MINIMUM_AVAILABLE_RAM_BYTES
        and swap_increase <= MAXIMUM_SWAP_INCREASE_BYTES
        and all(
            0
            <= int(item["cuda_peak_allocated_bytes"])
            <= int(item["cuda_peak_reserved_bytes"])
            <= int(item["cuda_total_memory_bytes"])
            for item in results
        )
    )
    summary = {
        "schema_version": 1,
        "phase": phase,
        "worker_count": worker_count,
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "worker_result_bindings": [
            {
                "path": str(candidate_root / f"worker-{index}.json"),
                "sha256": _file_sha256(candidate_root / f"worker-{index}.json"),
                "bytes": (candidate_root / f"worker-{index}.json").stat().st_size,
            }
            for index in range(worker_count)
        ],
        "synthetic_trace_sha256": next(iter(digests)) if len(digests) == 1 else None,
        "semantic_equivalence_passed": equivalent,
        "median_service_seconds": median_service,
        "maximum_service_seconds": max(service_times),
        "ready_excluded_makespan_seconds": makespan,
        "primary_capacity_work_units_per_second": capacity,
        "secondary_capacity_work_units_per_second": worker_count / makespan,
        "resource_envelope": resource_output,
        "swap_increase_bytes": swap_increase,
        "filesystem_available_bytes": shutil.disk_usage(output_root).free,
        "hard_admissible": hard_admissible,
        "quality_values_accessed": False,
    }
    _write_json_exclusive(candidate_root / "candidate.summary.json", summary)
    scheduler_lease.assert_held()
    device_guard_lease.assert_held()
    return summary


def _select_worker_count(phase_a: Sequence[Mapping[str, Any]]) -> int:
    _require(
        [item.get("worker_count") for item in phase_a] == list(CANDIDATE_WORKER_COUNTS)
        and all(item.get("quality_values_accessed") is False for item in phase_a),
        "Topology selector received a reordered or quality-observing phase A.",
    )
    admissible = [item for item in phase_a if item.get("hard_admissible") is True]
    _require(
        bool(admissible) and any(item.get("worker_count") == 1 for item in admissible),
        "Topology selector requires an admissible one-worker reference.",
    )
    best_capacity = max(
        float(item["primary_capacity_work_units_per_second"]) for item in admissible
    )
    selected = min(
        int(item["worker_count"])
        for item in admissible
        if float(item["primary_capacity_work_units_per_second"])
        >= NEAR_BEST_FRACTION * best_capacity
    )
    capacities = {
        int(item["worker_count"]): float(item["primary_capacity_work_units_per_second"])
        for item in admissible
    }
    if selected == 1 or capacities[selected] < MINIMUM_USEFUL_ACCELERATION * capacities[1]:
        return 1
    return selected


def _checkpoint_binding(
    *,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    lineage = legacy.load_superseded_empty_lineage_v1_3(
        trust_root=trust_root,
        repository_root=REPOSITORY_ROOT,
    )
    matches = [
        cast(Mapping[str, Any], row)
        for row in cast(list[Any], lineage.reuse_admission["calibrations"])
        if isinstance(row, Mapping)
        and row.get("scale") == SELECTED_SCALE
        and row.get("training_seed") == SELECTED_TRAINING_SEED
    ]
    _require(len(matches) == 1, "Probe checkpoint coordinate is not uniquely admitted.")
    binding = dict(cast(Mapping[str, Any], matches[0]["checkpoint"]))
    path = _absolute(Path(cast(str, binding["path"])))
    _require(
        path.is_file()
        and path.stat().st_size == binding["bytes"]
        and _file_sha256(path) == binding["sha256"],
        "Probe checkpoint differs from the exact admitted binding.",
    )
    return {**binding, "path": str(path)}


def _source_binding() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(not status, "Topology probe requires a clean tracked checkout.")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree_digest = contract.v1_3_5_implementation_tree_digest()
    _require(
        contract.v1_3_5_implementation_tree_digest_at_commit(source_commit) == tree_digest,
        "Topology probe source commit differs from its implementation tree.",
    )
    return {
        "source_commit": source_commit,
        "implementation_tree_digest": tree_digest,
    }


def run_probe(
    *,
    attestation_key_path: Path,
    output_root: Path = PROBE_OUTPUT_ROOT,
) -> dict[str, Any]:
    absolute_root = _absolute(output_root)
    _require(
        absolute_root == _absolute(PROBE_OUTPUT_ROOT) and not os.path.lexists(absolute_root),
        "Topology probe output must be the fresh canonical root.",
    )
    trust_root = attestation.load_trust_root(
        attestation_key_path,
        repository_root=REPOSITORY_ROOT,
        artifact_roots=(absolute_root,),
    )
    source = _source_binding()
    checkpoint = _checkpoint_binding(trust_root=trust_root)
    absolute_root.mkdir(parents=True, mode=0o700)
    os.chmod(absolute_root, 0o700)
    scheduler = acquire_gpu_lock(
        "p2-direct-controller-v1.3.5-topology-probe-supervisor",
        path=contract.DIRECT_GPU_SCHEDULER_LOCK_PATH,
    )
    guard = acquire_device_guard(
        "p2-direct-controller-v1.3.5-topology-probe-supervisor",
        _device_identity(),
    )
    try:
        phase_a: list[dict[str, Any]] = []
        reference: str | None = None
        for worker_count in CANDIDATE_WORKER_COUNTS:
            candidate = _run_candidate(
                phase="phase-a",
                worker_count=worker_count,
                prompt_tokens=PHASE_A_PROMPT_TOKENS,
                decode_tokens=PHASE_A_DECODE_TOKENS,
                checkpoint=Path(cast(str, checkpoint["path"])),
                output_root=absolute_root,
                scheduler_lease=scheduler,
                device_guard_lease=guard,
                reference_digest=reference,
            )
            if worker_count == 1:
                reference = cast(str, candidate["synthetic_trace_sha256"])
            phase_a.append(candidate)
        selected = _select_worker_count(phase_a)
        phase_b = _run_candidate(
            phase="phase-b",
            worker_count=selected,
            prompt_tokens=PHASE_B_PROMPT_TOKENS,
            decode_tokens=PHASE_B_DECODE_TOKENS,
            checkpoint=Path(cast(str, checkpoint["path"])),
            output_root=absolute_root,
            scheduler_lease=scheduler,
            device_guard_lease=guard,
            reference_digest=None,
        )
        semantic_passed = (
            all(item["semantic_equivalence_passed"] is True for item in phase_a)
            and phase_b["semantic_equivalence_passed"] is True
        )
        status = "GO" if semantic_passed and phase_b["hard_admissible"] is True else "NO-GO"
        payload = legacy._attested_payload(
            {
                "schema_version": PROBE_SCHEMA_VERSION,
                "experiment_id": PROBE_EXPERIMENT_ID,
                "status": status,
                "candidate_worker_counts": list(CANDIDATE_WORKER_COUNTS),
                "candidate_execution_order": list(CANDIDATE_WORKER_COUNTS),
                "selected_worker_count": selected,
                "selection_rule": {
                    "minimum_useful_acceleration_over_worker_count_1": (
                        MINIMUM_USEFUL_ACCELERATION
                    ),
                    "near_best_fraction": NEAR_BEST_FRACTION,
                    "tie_break": "smallest-worker-count",
                },
                "phase_a": phase_a,
                "phase_b": phase_b,
                "semantic_equivalence_passed": semantic_passed,
                "quality_values_accessed": False,
                "heldout_examples_materialized": 0,
                "targets_or_predictions_opened_by_selector": False,
                "checkpoint": checkpoint,
                "source": source,
                "selected_device_routing_identity": _device_identity(),
                "scheduler_lock": {
                    "path": str(scheduler.path),
                    "device": scheduler.device,
                    "inode": scheduler.inode,
                },
                "device_guard": {
                    "path": str(guard.path),
                    "device": guard.device,
                    "inode": guard.inode,
                },
                "v2_draft_relation": (
                    "prospective-v3-system-only-circular-authority-correction-"
                    "before-any-probe-or-v1.3.5-quality-access"
                ),
            },
            trust_root=trust_root,
            purpose=PROBE_ATTESTATION_PURPOSE,
        )
        _write_json_exclusive(absolute_root / PROBE_RECEIPT_PATH.name, payload)
        _require(status == "GO", "Topology probe failed its phase-B gate.")
        return payload
    finally:
        try:
            guard.close()
        finally:
            scheduler.close()


def load_probe_binding(
    path: Path,
    *,
    trust_root: attestation.TrustRoot,
) -> dict[str, Any]:
    absolute = _absolute(path)
    raw = absolute.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    _require(
        isinstance(payload, dict) and raw == legacy.canonical_pretty_json(payload),
        "Topology probe receipt bytes are not canonical.",
    )
    checked = cast(dict[str, Any], payload)
    legacy._verify_attested_payload(
        checked,
        trust_root=trust_root,
        purpose=PROBE_ATTESTATION_PURPOSE,
        label="V1.3.5 topology probe",
    )
    envelope = cast(Mapping[str, Any], checked["attestation"])
    _require(
        checked.get("status") == "GO"
        and checked.get("candidate_worker_counts") == list(CANDIDATE_WORKER_COUNTS)
        and checked.get("selected_worker_count") in CANDIDATE_WORKER_COUNTS
        and checked.get("semantic_equivalence_passed") is True
        and checked.get("quality_values_accessed") is False,
        "Topology probe is not admissible for v1.3.5.",
    )
    return {
        "path": str(absolute),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "payload_sha256": checked["payload_sha256"],
        "attestation_mac": envelope["mac"],
        "candidate_worker_counts": list(CANDIDATE_WORKER_COUNTS),
        "selected_worker_count": checked["selected_worker_count"],
        "semantic_equivalence_passed": True,
        "quality_values_accessed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v1.3.5 quality-blind same-GPU topology probe."
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-tokens", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--decode-tokens", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--ready-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--start-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--attestation-key-path", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROBE_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.child:
        _require(
            args.checkpoint is not None
            and args.ready_path is not None
            and args.start_path is not None
            and 0 <= args.worker_index < args.worker_count
            and args.worker_count in CANDIDATE_WORKER_COUNTS
            and args.prompt_tokens in {PHASE_A_PROMPT_TOKENS, PHASE_B_PROMPT_TOKENS}
            and args.decode_tokens in {PHASE_A_DECODE_TOKENS, PHASE_B_DECODE_TOKENS},
            "Topology probe child arguments are invalid.",
        )
        return _child_main(args)
    key_path = args.attestation_key_path
    if key_path is None:
        raw_key_path = os.environ.get(attestation.KEY_PATH_ENV)
        _require(bool(raw_key_path), "Topology probe attestation key path is required.")
        key_path = Path(cast(str, raw_key_path))
    payload = run_probe(attestation_key_path=key_path, output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_worker_count": payload["selected_worker_count"],
                "semantic_equivalence_passed": payload["semantic_equivalence_passed"],
                "quality_values_accessed": payload["quality_values_accessed"],
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
