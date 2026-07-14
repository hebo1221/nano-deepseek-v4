from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from adaptive_v4_gpu_lock import acquire_gpu_lock
from p3_natural_safety_workloads import generation_cases, rendered_case
from p3_sequence_gate import require_p3_sequence_gate
from prepare_p3_natural_safety_assets import (
    sha256,
    validate_ifeval_rows,
    validate_longsafety_rows,
    verify_file,
)
from run_p3_longbench_v2 import infer_one
from run_p3_mrcr import arm_config, atomic_json, failure_record, runtime_environment
from run_p3_ruler_matrix import KVPRESS_REVISION, git_dirty, git_head, load_evaluator
from validate_p3_natural_safety_manifest import validate_manifest
from verify_p3_natural_model import verify_snapshot

BENCHMARKS = ("LongSafety", "IFEval")
ARMS = ("native-dense", "strongest-memory-matched-fixed")
SEED = 9171402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def token_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        values = tensor.detach().cpu().to(torch.int64).contiguous().numpy()
        digest.update(int(values.shape[1]).to_bytes(8, "big"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def _parts(
    progress: Path, parts: Path, identity: dict[str, Any], expected: int
) -> list[dict[str, Any]]:
    if not progress.exists() and not parts.exists():
        atomic_json(progress, identity)
        parts.mkdir(parents=True)
        return []
    if progress.is_file() and not parts.exists():
        parts.mkdir(parents=True)
    if parts.is_dir() and not any(parts.iterdir()) and not progress.exists():
        atomic_json(progress, identity)
    _require(progress.is_file() and parts.is_dir(), "Natural safety partial state is incomplete.")
    _require(json.loads(progress.read_text()) == identity, "Natural safety provenance drifted.")
    files = sorted(parts.glob("*.json"))
    _require(
        [path.name for path in files] == [f"{index:06d}.json" for index in range(len(files))],
        "Natural safety partial sequence drifted.",
    )
    _require(len(files) <= expected, "Natural safety partial state has too many records.")
    return [json.loads(path.read_text()) for path in files]


def _completed(path: Path, arm: str, benchmark: str, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    _require(
        payload.get("experiment_id") == "p3-natural-safety-generation-arm-cell-v1"
        and payload.get("benchmark") == benchmark
        and payload.get("arm") == arm
        and payload.get("status") == "terminal"
        and payload.get("run_identity") == identity,
        f"Completed natural safety cell provenance drifted: {path}.",
    )
    records = Path(payload.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and payload["raw_records"]["sha256"] == sha256(records),
        f"Natural safety records drifted: {records}.",
    )
    return True


def _load_rows(
    benchmark: str, contract: dict[str, Any], inventory: dict[str, Any]
) -> list[dict[str, Any]]:
    observed = inventory["benchmarks"][benchmark]["dataset"]
    _require(
        observed["repo_id"] == contract["dataset"]["repo_id"]
        and observed["revision"] == contract["dataset"]["revision"]
        and observed["license"] == contract["dataset"]["license"],
        f"{benchmark} dataset inventory drifted.",
    )
    row_entry = next(entry for entry in contract["dataset"]["files"] if "rows" in entry)
    file_metadata = next(
        entry for entry in observed["files"] if Path(entry["path"]).name == row_entry["path"]
    )
    path = Path(file_metadata["path"])
    verify_file(path, row_entry)
    if benchmark == "LongSafety":
        payload = json.loads(path.read_text())
        validate_longsafety_rows(
            payload,
            expected_rows=contract["prompt_protocol"]["expected_rows"],
            required_fields=contract["dataset"]["required_fields"],
        )
    else:
        payload = [json.loads(line) for line in path.read_text().splitlines() if line]
        validate_ifeval_rows(
            payload,
            expected_rows=contract["protocol"]["expected_prompts_per_arm"],
            required_fields=contract["dataset"]["required_fields"],
        )
    return payload


def _failure(
    base: dict[str, Any], failure_type: str, started: int, error: BaseException | None = None
) -> dict[str, Any]:
    return failure_record(
        base,
        failure_type=failure_type,
        latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
        peak_hbm_bytes=torch.cuda.max_memory_allocated(),
        error=error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen P3 natural safety generations.")
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--kvpress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-safety-v1.json"),
    )
    parser.add_argument(
        "--natural-manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--asset-inventory",
        type=Path,
        default=Path(
            "artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety/assets/inventory.json"
        ),
    )
    parser.add_argument(
        "--fixed-selection",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/fixed-baseline-selection.json"),
    )
    parser.add_argument(
        "--p2-matrix",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-core-quality-matrix.json"),
    )
    parser.add_argument(
        "--causal-gate",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p2-causal-ablation.summary.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural-safety"),
    )
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--max-new-examples", type=int)
    args = parser.parse_args()
    _require(
        args.max_new_examples is None or args.max_new_examples > 0,
        "max-new-examples must be positive.",
    )
    sequence = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source_commit = git_head(Path.cwd())
    _require(not git_dirty(Path.cwd()), "Natural safety generation requires a clean tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its frozen clean revision.",
    )
    raw_manifest = args.manifest.read_bytes()
    manifest = json.loads(raw_manifest)
    validation = validate_manifest(manifest)
    natural = json.loads(args.natural_manifest.read_text())
    _require(
        natural["model"]["revision"] == manifest["model"]["revision"]
        and natural["model"]["snapshot_digest_set_sha256"]
        == manifest["model"]["snapshot_digest_set_sha256"],
        "Natural safety model dependency drifted.",
    )
    snapshot = args.model_snapshot.resolve()
    verify_snapshot(snapshot, natural["model"])
    inventory = json.loads(args.asset_inventory.read_text())
    _require(
        inventory.get("experiment_id") == "p3-natural-safety-asset-inventory-v1"
        and inventory.get("status") == "verified"
        and inventory.get("source", {}).get("dirty") is False
        and inventory.get("manifest", {}).get("sha256") == hashlib.sha256(raw_manifest).hexdigest(),
        "Natural safety asset inventory is not a clean matching artifact.",
    )
    selection = json.loads(args.fixed_selection.read_text())
    _require(
        selection.get("experiment_id") == "p3-fixed-baseline-selection-v1"
        and selection.get("source", {}).get("dirty") is False,
        "Natural safety fixed baseline selection is invalid.",
    )
    benchmark = args.benchmark
    contract = manifest["benchmarks"][benchmark]
    rows = _load_rows(benchmark, contract, inventory)
    cases = generation_cases(benchmark, rows)
    expected = (
        contract["prompt_protocol"]["expected_predictions_per_arm"]
        if benchmark == "LongSafety"
        else contract["protocol"]["expected_prompts_per_arm"]
    )
    _require(len(cases) == expected, f"{benchmark} generation grid drifted.")
    generation_reserve = (
        contract["prompt_protocol"]["generation_max_new_tokens"]
        if benchmark == "LongSafety"
        else contract["protocol"]["generation_max_new_tokens"]
    )
    manifest_digest = hashlib.sha256(raw_manifest).hexdigest()
    inventory_digest = sha256(args.asset_inventory)
    natural_digest = sha256(args.natural_manifest)
    selection_digest = sha256(args.fixed_selection)
    runner_digest = sha256(Path(__file__))
    selected_arms = tuple(args.arm or ARMS)
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "manifest_sha256": manifest_digest,
            "natural_manifest_sha256": natural_digest,
            "asset_inventory_sha256": inventory_digest,
            "fixed_selection_sha256": selection_digest,
            "model_snapshot_digest_set_sha256": manifest["model"][
                "snapshot_digest_set_sha256"
            ],
            "benchmark": benchmark,
            "arm_config": arm_config(arm, selection, selection_digest),
            "seed": SEED,
        }
        for arm in selected_arms
    }
    root = args.output_root / benchmark.lower()
    pending = [
        arm
        for arm in selected_arms
        if not _completed(root / arm / "cell.json", arm, benchmark, identities[arm])
    ]
    if not pending:
        print(json.dumps({"benchmark": benchmark, "status": "complete"}))
        return
    _require(torch.cuda.is_available(), "Natural safety generation requires CUDA.")
    lock = acquire_gpu_lock(f"p3-natural-safety-{benchmark.lower()}")
    try:
        EvaluationConfig, EvaluationRunner, _unused = load_evaluator(kvpress_root)
        config = EvaluationConfig(
            dataset=f"natural-safety-{benchmark.lower()}",
            model=str(snapshot),
            device="cuda:0",
            press_name="no_press",
            compression_ratio=0.0,
            output_dir=str(root),
            seed=SEED,
            max_context_length=manifest["model"]["maximum_supported_context_tokens"],
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        runner = EvaluationRunner(config)
        runner._setup_press()
        runner._setup_model_pipeline()
        tokenizer = runner.pipeline.tokenizer
        environment = runtime_environment()
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        for arm in pending:
            settings = identities[arm]["arm_config"]
            runner.config.press_name = settings["press_name"]
            runner.config.compression_ratio = settings["compression_ratio"]
            runner._setup_press()
            arm_root = root / arm
            progress = arm_root / "progress.json"
            parts = arm_root / "record-parts"
            existing = _parts(progress, parts, identities[arm], expected)
            _require(
                all(existing[index].get("example_id") == cases[index].example_id for index in range(len(existing))),
                "Natural safety resume coordinate order drifted.",
            )
            limit = expected
            if args.max_new_examples is not None:
                limit = min(expected, len(existing) + args.max_new_examples)
            arm_root.mkdir(parents=True, exist_ok=True)
            for position in range(len(existing), limit):
                case = cases[position]
                rendered = rendered_case(tokenizer, case)
                base = {
                    "example_id": case.example_id,
                    "source_id": case.source_id,
                    "benchmark": benchmark,
                    "arm": arm,
                    "prompt_position": case.prompt_position,
                    "exact_input_tokens": rendered["exact_input_tokens"],
                    "generation_reserve_tokens": generation_reserve,
                    "raw_prompt_sha256": rendered["raw_prompt_sha256"],
                    "input_token_ids_sha256": token_digest(
                        rendered["context_ids"], rendered["question_ids"]
                    ),
                    "token_boundary_retreat": rendered["token_boundary_retreat"],
                    "arm_config": settings,
                    "metadata": case.metadata,
                }
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                if rendered["exact_input_tokens"] + generation_reserve > manifest["model"][
                    "maximum_supported_context_tokens"
                ]:
                    record = _failure(base, "unsupported-context", started)
                else:
                    response: str | None = None
                    resident = 0
                    try:
                        response, resident = infer_one(
                            pipeline=runner.pipeline,
                            press=runner.press,
                            rendered=rendered,
                            max_new_tokens=generation_reserve,
                        )
                        torch.cuda.synchronize()
                        if not response.strip():
                            record = _failure(base, "empty-generation", started)
                        else:
                            generated = len(tokenizer.encode(response, add_special_tokens=False))
                            record = {
                                **base,
                                "status": "generated",
                                "raw_response": response,
                                "score": None,
                                "evaluation_status": (
                                    "pending-paid-official-judge"
                                    if benchmark == "LongSafety"
                                    else "pending-official-deterministic-scorer"
                                ),
                                "failure_type": None,
                                "stop_reason": (
                                    "max-new-tokens"
                                    if generated >= generation_reserve
                                    else "eos-or-special-token"
                                ),
                                "generated_tokens_observed": generated,
                                "latency_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                                "peak_hbm_bytes": torch.cuda.max_memory_allocated(),
                                "hot_resident_bytes": resident,
                                "error_type": None,
                                "error": None,
                            }
                    except torch.cuda.OutOfMemoryError as error:
                        record = _failure(base, "oom", started, error)
                        torch.cuda.empty_cache()
                    except Exception as error:
                        record = _failure(base, "runtime-error", started, error)
                atomic_json(parts / f"{position:06d}.json", record)
                print(
                    json.dumps(
                        {
                            "benchmark": benchmark,
                            "arm": arm,
                            "completed_examples": position + 1,
                            "total": expected,
                        }
                    ),
                    flush=True,
                )
            if limit < expected:
                continue
            completed = _parts(progress, parts, identities[arm], expected)
            _require(len(completed) == expected, "Natural safety generation grid did not close.")
            records = arm_root / "records.jsonl"
            temporary = arm_root / ".records.jsonl.tmp"
            temporary.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in completed)
            )
            temporary.replace(records)
            cell = {
                "schema_version": 1,
                "experiment_id": "p3-natural-safety-generation-arm-cell-v1",
                "benchmark": benchmark,
                "arm": arm,
                "status": "terminal",
                "source": {"commit": source_commit, "dirty": False},
                "run_identity": identities[arm],
                "manifest": {"path": str(args.manifest), "sha256": manifest_digest},
                "manifest_validation": validation,
                "asset_inventory": {"path": str(args.asset_inventory), "sha256": inventory_digest},
                "natural_manifest": {"path": str(args.natural_manifest), "sha256": natural_digest},
                "fixed_selection": {"path": str(args.fixed_selection), "sha256": selection_digest},
                "sequence_gate": sequence,
                "environment": environment,
                "expected_generations": expected,
                "raw_records": {"path": str(records), "sha256": sha256(records)},
            }
            atomic_json(arm_root / "cell.json", cell)
            progress.unlink()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
