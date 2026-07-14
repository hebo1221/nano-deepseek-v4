from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from p3_sequence_gate import require_p3_sequence_gate
from prepare_p3_ruler_dataset import (
    RULER_REVISION,
    SOURCE_FILES,
    TASKS,
    git_head,
    load_rows,
    run_upstream_task,
    sha256,
)
from run_p3_ruler_matrix import git_dirty, split_prompt
from transformers import AutoTokenizer
from validate_p3_natural_suite_manifest import validate_manifest
from verify_p3_natural_model import verify_snapshot

MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
LENGTHS = (8192, 16384, 32768, 65536, 131072)
SAMPLES_PER_TASK = 500
EXPECTED_ROWS_PER_LENGTH = len(TASKS) * SAMPLES_PER_TASK


def workspace_source() -> dict[str, Any]:
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {
        "commit": git_head(Path.cwd()),
        "dirty": dirty,
        "implementation_sha256": sha256(Path(__file__)),
    }


def inspect_task_rows(
    *, path: Path, task: str, tokenizer: Any, expected_rows: int, ceiling: int
) -> dict[str, Any]:
    rows = load_rows(path, expected_rows)
    token_counts: list[int] = []
    identity_digests: list[str] = []
    for row_number, row in enumerate(rows):
        split_prompt(row["input"], task)
        token_count = len(tokenizer.encode(row["input"], add_special_tokens=False))
        if token_count > ceiling:
            raise ValueError(f"{path}:{row_number + 1} exceeds its {ceiling}-token ceiling.")
        token_counts.append(token_count)
        identity = json.dumps(
            {"input": row["input"], "outputs": row["outputs"]},
            sort_keys=True,
            separators=(",", ":"),
        )
        identity_digests.append(hashlib.sha256(identity.encode()).hexdigest())
    if len(set(identity_digests)) != len(identity_digests):
        raise ValueError(f"RULER task contains duplicate examples: {path}.")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "rows": len(rows),
        "token_count_min": min(token_counts),
        "token_count_max": max(token_counts),
        "example_identity_set_sha256": hashlib.sha256(
            "\n".join(sorted(identity_digests)).encode()
        ).hexdigest(),
    }


def prepare_length(
    *,
    ruler_root: Path,
    tokenizer_path: Path,
    output_root: Path,
    tokenizer: Any,
    length: int,
    samples: int,
    seed: int,
    source: dict[str, Any],
    source_files: dict[str, dict[str, Any]],
    natural_manifest: dict[str, Any],
    natural_manifest_path: Path,
) -> None:
    output = output_root / str(length)
    task_artifacts: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        path = output / task / "validation.jsonl"
        try:
            artifact = inspect_task_rows(
                path=path,
                task=task,
                tokenizer=tokenizer,
                expected_rows=samples,
                ceiling=length,
            )
        except (FileNotFoundError, ValueError):
            run_upstream_task(ruler_root, tokenizer_path, output, length, task, samples, seed)
            artifact = inspect_task_rows(
                path=path,
                task=task,
                tokenizer=tokenizer,
                expected_rows=samples,
                ceiling=length,
            )
        task_artifacts[task] = artifact
        print(
            json.dumps({"length": length, "task": task, "max_tokens": artifact["token_count_max"]}),
            flush=True,
        )

    total_rows = sum(artifact["rows"] for artifact in task_artifacts.values())
    if total_rows != len(TASKS) * samples:
        raise ValueError(f"RULER {length} total row count drifted: {total_rows}.")
    payload = {
        "schema_version": 1,
        "experiment_id": "p3-natural-ruler-qwen3-4b-dataset-v1",
        "source": source,
        "natural_suite_manifest": {
            "path": str(natural_manifest_path.resolve()),
            "sha256": sha256(natural_manifest_path),
        },
        "ruler": {
            "revision": RULER_REVISION,
            "path": str(ruler_root),
            "clean_tracked_tree": True,
        },
        "tokenizer": {
            "model_revision": MODEL_REVISION,
            "path": str(tokenizer_path),
            "snapshot_digest_set_sha256": natural_manifest["model"]["snapshot_digest_set_sha256"],
            "tokenizer_config_sha256": sha256(tokenizer_path / "tokenizer_config.json"),
        },
        "generation": {
            "length_tokens": length,
            "samples_per_task": samples,
            "random_seed": seed,
            "tasks": list(TASKS),
            "total_rows": total_rows,
            "observed_token_count_min": min(
                artifact["token_count_min"] for artifact in task_artifacts.values()
            ),
            "observed_token_count_max": max(
                artifact["token_count_max"] for artifact in task_artifacts.values()
            ),
        },
        "source_files": source_files,
        "task_artifacts": task_artifacts,
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "dataset-manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate frozen Qwen3-4B RULER data.")
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p3-natural-suite-v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/natural/ruler-qwen3-4b/data"),
    )
    parser.add_argument("--length", type=int, action="append", choices=LENGTHS)
    parser.add_argument("--samples-per-task", type=int, default=SAMPLES_PER_TASK)
    parser.add_argument("--seed", type=int, default=42)
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
    args = parser.parse_args()
    require_p3_sequence_gate(args.p2_matrix, args.causal_gate)
    if args.samples_per_task != SAMPLES_PER_TASK:
        raise ValueError("Natural RULER requires exactly 500 samples per task.")
    source = workspace_source()
    if source["dirty"]:
        raise RuntimeError("Natural RULER generation requires a clean source tree.")

    ruler_root = args.ruler_root.resolve()
    tokenizer_path = args.tokenizer_snapshot.resolve()
    if git_head(ruler_root) != RULER_REVISION or git_dirty(ruler_root):
        raise ValueError("RULER checkout is not at its clean frozen revision.")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if manifest["model"]["revision"] != MODEL_REVISION:
        raise ValueError("Natural RULER model revision drifted.")
    verify_snapshot(tokenizer_path, manifest["model"])
    ruler_contract = manifest["benchmarks"]["RULER"]
    if tuple(ruler_contract["lengths_tokens"]) != LENGTHS:
        raise ValueError("Natural RULER length grid drifted.")
    if ruler_contract["samples_per_task"] != SAMPLES_PER_TASK:
        raise ValueError("Natural RULER sample count drifted.")

    source_dir = ruler_root / "scripts" / "data" / "synthetic" / "json"
    source_files: dict[str, dict[str, Any]] = {}
    for name in SOURCE_FILES:
        path = source_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing RULER source file: {path}")
        source_files[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    for length in tuple(args.length or LENGTHS):
        prepare_length(
            ruler_root=ruler_root,
            tokenizer_path=tokenizer_path,
            output_root=args.output_root.resolve(),
            tokenizer=tokenizer,
            length=length,
            samples=args.samples_per_task,
            seed=args.seed,
            source=source,
            source_files=source_files,
            natural_manifest=manifest,
            natural_manifest_path=args.manifest,
        )


if __name__ == "__main__":
    main()
