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
from p3_natural_workloads import (
    encode_rendered_system_context_query_exact,
    render_chat_split_system_context_query,
)
from p3_safety_workloads import FAMILIES, build_example, score_response
from p3_sequence_gate import require_p3_sequence_gate
from run_p3_longbench_v2 import infer_one
from run_p3_mrcr import atomic_json, failure_record, runtime_environment
from run_p3_ruler_matrix import KVPRESS_REVISION, git_dirty, git_head, load_evaluator
from verify_p3_natural_model import sha256, verify_snapshot

BENCHMARK = "SafetyStress"
ARMS = (
    "native-dense",
    "strongest-memory-matched-fixed",
    "strongest-memory-matched-fixed+protected-prefix",
)


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


def failure_safety_fields(example: Any, response: str | None) -> dict[str, bool]:
    if response:
        scores = score_response(example, response)
        return {
            "exact_required_response": bool(scores["exact_required_response"]),
            "leakage_event": bool(scores["leakage_event"]),
        }
    return {"exact_required_response": False, "leakage_event": False}


def coordinates(manifest: dict[str, Any]) -> list[tuple[str, int, int]]:
    rows = [
        (family, context, index)
        for family in manifest["families"]
        for context in manifest["context_targets_tokens"]
        for index in range(manifest["examples_per_family_context"])
    ]
    _require(
        len(rows) == manifest["expected_examples_per_arm"],
        "Safety stress Cartesian total drifted.",
    )
    return rows


def safety_arm_config(
    arm: str, selection: dict[str, Any], selection_digest: str
) -> dict[str, Any]:
    if arm == "native-dense":
        return {
            "press_name": "no_press",
            "compression_ratio": 0.0,
            "protected_prefix": False,
        }
    _require(arm in ARMS[1:], f"Unknown safety arm: {arm}")
    return {
        "press_name": selection["selected_arm"],
        "compression_ratio": selection["selected_compression_ratio"],
        "selection_sha256": selection_digest,
        "protected_prefix": arm.endswith("+protected-prefix"),
    }


def load_contracts(
    *, safety_manifest_path: Path, natural_manifest_path: Path, selection_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    safety = json.loads(safety_manifest_path.read_text())
    _require(
        safety.get("experiment_id") == "p3-qwen3-4b-safety-stress-v1"
        and safety.get("status") == "frozen_before_execution"
        and tuple(safety.get("arms", ())) == ARMS
        and tuple(safety.get("families", {})) == FAMILIES
        and safety.get("expected_examples_per_arm") == 1200,
        "Frozen safety stress manifest drifted.",
    )
    natural = json.loads(natural_manifest_path.read_text())
    _require(
        natural["model"]["revision"] == safety["model"]["revision"],
        "Safety and natural model revisions diverged.",
    )
    selection = json.loads(selection_path.read_text())
    _require(
        selection.get("experiment_id") == "p3-fixed-baseline-selection-v1"
        and selection.get("source", {}).get("dirty") is False
        and selection.get("selected_arm")
        in {
            "streaming_llm",
            "snapkv",
            "pyramidkv",
            "adakv_snapkv",
            "expected_attention",
            "critical_expected_attention",
        }
        and selection.get("selected_compression_ratio") == 0.5,
        "Frozen fixed baseline selection is missing or invalid.",
    )
    return safety, natural, selection


def _existing_parts(
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
    _require(progress.is_file() and parts.is_dir(), "Partial safety state is incomplete.")
    _require(json.loads(progress.read_text()) == identity, "Partial safety provenance drifted.")
    files = sorted(parts.glob("*.json"))
    _require(
        [path.name for path in files] == [f"{index:06d}.json" for index in range(len(files))],
        "Partial safety example sequence drifted.",
    )
    records = [json.loads(path.read_text()) for path in files]
    _require(len(records) <= expected, "Partial safety state has too many records.")
    return records


def _completed(path: Path, arm: str, identity: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text())
    _require(
        payload.get("experiment_id") == "p3-safety-stress-arm-cell-v1"
        and payload.get("benchmark") == BENCHMARK
        and payload.get("arm") == arm
        and payload.get("status") == "terminal"
        and payload.get("run_identity") == identity,
        f"Completed safety arm provenance drifted: {path}.",
    )
    records = Path(payload.get("raw_records", {}).get("path", ""))
    _require(
        records.is_file() and payload["raw_records"]["sha256"] == sha256(records),
        f"Completed safety records drifted: {records}.",
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Qwen3-4B safety stress arms.")
    parser.add_argument("--kvpress-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-safety-stress-v1.json"),
    )
    parser.add_argument(
        "--natural-manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
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
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/safety-stress"),
    )
    parser.add_argument("--arm", action="append", choices=ARMS)
    parser.add_argument("--max-new-examples", type=int)
    args = parser.parse_args()
    _require(
        args.max_new_examples is None or args.max_new_examples > 0,
        "max-new-examples must be positive.",
    )
    sequence_decision = require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    source_commit = git_head(Path.cwd())
    _require(not git_dirty(Path.cwd()), "Safety stress requires a clean source tree.")
    kvpress_root = args.kvpress_root.resolve()
    _require(
        git_head(kvpress_root) == KVPRESS_REVISION and not git_dirty(kvpress_root),
        "KVPress checkout does not match its clean frozen revision.",
    )
    safety, natural, selection = load_contracts(
        safety_manifest_path=args.manifest,
        natural_manifest_path=args.natural_manifest,
        selection_path=args.fixed_selection,
    )
    snapshot = args.model_snapshot.resolve()
    verify_snapshot(snapshot, natural["model"])
    runner_digest = sha256(Path(__file__))
    workload_digest = sha256(Path(__file__).with_name("p3_safety_workloads.py"))
    manifest_digest = sha256(args.manifest)
    natural_manifest_digest = sha256(args.natural_manifest)
    selection_digest = sha256(args.fixed_selection)
    causal_digest = sha256(args.causal_gate)
    expected = safety["expected_examples_per_arm"]
    grid = coordinates(safety)
    selected_arms = tuple(args.arm or ARMS)
    identities = {
        arm: {
            "source_commit": source_commit,
            "implementation_sha256": runner_digest,
            "workload_sha256": workload_digest,
            "manifest_sha256": manifest_digest,
            "natural_manifest_sha256": natural_manifest_digest,
            "fixed_selection_sha256": selection_digest,
            "causal_gate_sha256": causal_digest,
            "model_snapshot_digest_set_sha256": natural["model"][
                "snapshot_digest_set_sha256"
            ],
            "arm_config": safety_arm_config(arm, selection, selection_digest),
            "seed": safety["seed"],
        }
        for arm in selected_arms
    }
    pending = [
        arm
        for arm in selected_arms
        if not _completed(args.output_root / arm / "cell.json", arm, identities[arm])
    ]
    if not pending:
        print(json.dumps({"status": "complete", "arms": list(selected_arms)}))
        return
    _require(torch.cuda.is_available(), "Safety stress requires CUDA.")
    lock = acquire_gpu_lock("p3-safety-stress")
    try:
        EvaluationConfig, EvaluationRunner, _unused = load_evaluator(kvpress_root)
        from p3_protected_prefix_press import wrap_same_budget_protected_prefix

        config = EvaluationConfig(
            dataset="safety-stress",
            model=str(snapshot),
            device="cuda:0",
            press_name="no_press",
            compression_ratio=0.0,
            output_dir=str(args.output_root),
            seed=safety["seed"],
            max_context_length=safety["model"]["maximum_supported_context_tokens"],
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        runner = EvaluationRunner(config)
        runner._setup_press()
        runner._setup_model_pipeline()
        tokenizer = runner.pipeline.tokenizer
        environment = runtime_environment()
        random.seed(safety["seed"])
        np.random.seed(safety["seed"])
        torch.manual_seed(safety["seed"])
        torch.cuda.manual_seed_all(safety["seed"])
        reserve = safety["generation_reserve_tokens"]
        minimum_fraction = safety["target_fill_tolerance"]["minimum_fraction"]
        revisions = {
            "model_revision": safety["model"]["revision"],
            "workload_sha256": workload_digest,
            "scorer_sha256": workload_digest,
        }
        for arm in pending:
            settings = identities[arm]["arm_config"]
            runner.config.press_name = settings["press_name"]
            runner.config.compression_ratio = settings["compression_ratio"]
            runner._setup_press()
            active_press: Any = runner.press
            protected_press: Any = None
            if settings["protected_prefix"]:
                protected_press = wrap_same_budget_protected_prefix(runner.press)
                active_press = protected_press
            root = args.output_root / arm
            progress = root / "progress.json"
            parts = root / "record-parts"
            existing = _existing_parts(progress, parts, identities[arm], expected)
            _require(
                all(
                    existing[index]["coordinates"]
                    == {"family": family, "context_target": context, "index": example_index}
                    for index, (family, context, example_index) in enumerate(grid[: len(existing)])
                ),
                "Partial safety coordinate order drifted.",
            )
            limit = expected
            if args.max_new_examples is not None:
                limit = min(limit, len(existing) + args.max_new_examples)
            root.mkdir(parents=True, exist_ok=True)
            for position in range(len(existing), limit):
                family, context_target, example_index = grid[position]
                example = build_example(
                    tokenizer=tokenizer,
                    seed=safety["seed"],
                    family=family,
                    context_target=context_target,
                    index=example_index,
                    generation_reserve=reserve,
                    minimum_fraction=minimum_fraction,
                )
                rendered_prefix, rendered_context, rendered_query = (
                    render_chat_split_system_context_query(
                        tokenizer,
                        example.system_prefix,
                        example.context,
                        example.query,
                    )
                )
                (
                    context_ids,
                    query_ids,
                    protected_length,
                    protected_retreat,
                    query_retreat,
                ) = encode_rendered_system_context_query_exact(
                    tokenizer, rendered_prefix, rendered_context, rendered_query
                )
                exact_tokens = int(context_ids.shape[1] + query_ids.shape[1])
                _require(exact_tokens == example.exact_input_tokens, "Safety token count drifted.")
                base = {
                    "example_id": example.example_id,
                    "benchmark": BENCHMARK,
                    "arm": arm,
                    "coordinates": {
                        "family": family,
                        "context_target": context_target,
                        "index": example_index,
                    },
                    "family": family,
                    "context_target": context_target,
                    "exact_input_tokens": exact_tokens,
                    "generation_reserve_tokens": reserve,
                    "raw_prompt_sha256": hashlib.sha256(
                        (rendered_prefix + rendered_context + rendered_query).encode()
                    ).hexdigest(),
                    "input_token_ids_sha256": token_digest(context_ids, query_ids),
                    "token_boundary_retreat": query_retreat,
                    "query_boundary_retreat": query_retreat,
                    "protected_prefix_token_span": {
                        "start": 0,
                        "end": protected_length,
                        "tokens": protected_length,
                        "stable_boundary_retreat": protected_retreat,
                    },
                    "expected_response_sha256": hashlib.sha256(example.expected.encode()).hexdigest(),
                    "canary_sha256": (
                        hashlib.sha256(example.canary.encode()).hexdigest()
                        if example.canary is not None
                        else None
                    ),
                    "arm_config": settings,
                    "revisions": revisions,
                }
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                response: str | None = None
                resident = 0
                try:
                    if protected_press is not None:
                        protected_press.configure(
                            protected_start=0, protected_end=protected_length
                        )
                    response, resident = infer_one(
                        pipeline=runner.pipeline,
                        press=active_press,
                        rendered={"context_ids": context_ids, "question_ids": query_ids},
                        max_new_tokens=reserve,
                    )
                    physical_audit = (
                        protected_press.audit() if protected_press is not None else None
                    )
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
                    peak = torch.cuda.max_memory_allocated()
                    generated = len(tokenizer.encode(response, add_special_tokens=False))
                    if not response.strip():
                        record = {
                            **failure_record(
                                base,
                                failure_type="empty-generation",
                                latency_ms=elapsed,
                                peak_hbm_bytes=peak,
                                hot_resident_bytes=resident,
                            ),
                            **failure_safety_fields(example, response),
                        }
                    else:
                        scores = score_response(example, response)
                        record = {
                            **base,
                            "status": "scored",
                            "raw_response": response,
                            "parsed_response": scores["normalized_response"],
                            "score": scores["score"],
                            "exact_required_response": scores["exact_required_response"],
                            "leakage_event": scores["leakage_event"],
                            "failure_type": None,
                            "stop_reason": (
                                "max-new-tokens" if generated >= reserve else "eos-or-special-token"
                            ),
                            "generated_tokens_observed": generated,
                            "latency_ms": elapsed,
                            "peak_hbm_bytes": peak,
                            "hot_resident_bytes": resident,
                            "protected_prefix_physical_audit": physical_audit,
                            "error_type": None,
                            "error": None,
                        }
                except torch.cuda.OutOfMemoryError as error:
                    record = {
                        **failure_record(
                            base,
                            failure_type="oom",
                            latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                            peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                            error=error,
                        ),
                        **failure_safety_fields(example, response),
                    }
                    torch.cuda.empty_cache()
                except Exception as error:
                    record = {
                        **failure_record(
                            base,
                            failure_type="runtime-error",
                            latency_ms=(time.perf_counter_ns() - started) / 1_000_000.0,
                            peak_hbm_bytes=torch.cuda.max_memory_allocated(),
                            hot_resident_bytes=resident,
                            raw_response=response or "",
                            parsed_response=response,
                            error=error,
                        ),
                        **failure_safety_fields(example, response),
                    }
                atomic_json(parts / f"{position:06d}.json", record)
                print(
                    json.dumps({"arm": arm, "completed_examples": position + 1, "total": expected}),
                    flush=True,
                )
            if limit < expected:
                continue
            completed = _existing_parts(progress, parts, identities[arm], expected)
            _require(len(completed) == expected, "Safety record grid did not close.")
            records = root / "records.jsonl"
            temporary = root / ".records.jsonl.tmp"
            temporary.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in completed)
            )
            temporary.replace(records)
            cell = {
                "schema_version": 1,
                "experiment_id": "p3-safety-stress-arm-cell-v1",
                "benchmark": BENCHMARK,
                "arm": arm,
                "status": "terminal",
                "source": {"commit": source_commit, "dirty": False, "implementation_sha256": runner_digest},
                "run_identity": identities[arm],
                "manifest": {"path": str(args.manifest), "sha256": manifest_digest},
                "natural_manifest": {"path": str(args.natural_manifest), "sha256": natural_manifest_digest},
                "causal_gate": {"path": str(args.causal_gate), "sha256": causal_digest},
                "fixed_baseline_selection": {"path": str(args.fixed_selection), "sha256": selection_digest},
                "model_snapshot_digest_set_sha256": natural["model"]["snapshot_digest_set_sha256"],
                "p3_sequence_decision": sequence_decision,
                "environment": environment,
                "raw_records": {"path": str(records), "sha256": sha256(records)},
            }
            atomic_json(root / "cell.json", cell)
            progress.unlink()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
