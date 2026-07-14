from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

RULER_REVISION = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
LENGTHS = (8192, 16384, 32768)
TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
SOURCE_FILES = ("PaulGrahamEssays.json", "squad.json", "hotpotqa.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def workspace_source() -> dict[str, Any]:
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    return {"commit": git_head(Path.cwd()), "dirty": dirty}


def load_rows(path: Path, expected: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}.") from error
            if not isinstance(row.get("input"), str) or not isinstance(row.get("outputs"), list):
                raise ValueError(f"Invalid RULER row at {path}:{line_number}.")
            rows.append(row)
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows in {path}, found {len(rows)}.")
    return rows


def run_upstream_task(
    ruler_root: Path,
    tokenizer_path: Path,
    output: Path,
    length: int,
    task: str,
    samples: int,
    seed: int,
) -> None:
    command = [
        sys.executable,
        "data/prepare.py",
        "--save_dir",
        str(output),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--tokenizer_path",
        str(tokenizer_path),
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(length),
        "--model_template_type",
        "base",
        "--num_samples",
        str(samples),
        "--random_seed",
        str(seed),
    ]
    env = dict(os.environ)
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    completed = subprocess.run(
        command,
        cwd=ruler_root / "scripts",
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"RULER generator failed for length={length}, task={task}:\n{completed.stderr}"
        )
    print(completed.stdout, end="", flush=True)


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
) -> None:
    output = output_root / str(length)
    artifacts: dict[str, dict[str, Any]] = {}
    observed_min: int | None = None
    observed_max = 0
    for task in TASKS:
        path = output / task / "validation.jsonl"
        try:
            rows = load_rows(path, samples)
        except (FileNotFoundError, ValueError):
            run_upstream_task(ruler_root, tokenizer_path, output, length, task, samples, seed)
            rows = load_rows(path, samples)
        counts = [len(tokenizer.tokenize(row["input"])) for row in rows]
        task_min, task_max = min(counts), max(counts)
        if task_max > length:
            raise ValueError(f"{path} exceeds its {length}-token ceiling: {task_max}.")
        observed_min = task_min if observed_min is None else min(observed_min, task_min)
        observed_max = max(observed_max, task_max)
        artifacts[task] = {
            "path": str(path),
            "sha256": sha256(path),
            "rows": len(rows),
            "token_count_min": task_min,
            "token_count_max": task_max,
        }
        print(json.dumps({"length": length, "task": task, "max_tokens": task_max}), flush=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": "p3-ruler-qwen3-1.7b-dataset-v1",
        "source": source,
        "ruler": {"revision": RULER_REVISION, "path": str(ruler_root)},
        "tokenizer": {
            "model_revision": MODEL_REVISION,
            "path": str(tokenizer_path),
            "tokenizer_config_sha256": sha256(tokenizer_path / "tokenizer_config.json"),
        },
        "generation": {
            "length_tokens": length,
            "samples_per_task": samples,
            "random_seed": seed,
            "tasks": list(TASKS),
            "total_rows": samples * len(TASKS),
            "observed_token_count_min": observed_min,
            "observed_token_count_max": observed_max,
        },
        "source_files": source_files,
        "task_artifacts": artifacts,
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "dataset-manifest.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Qwen-tokenized frozen RULER data.")
    parser.add_argument("--ruler-root", type=Path, required=True)
    parser.add_argument("--tokenizer-snapshot", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/adaptive_v4_memory/paper_grade/p3/ruler-qwen3-1.7b/data"),
    )
    parser.add_argument("--length", type=int, action="append", choices=LENGTHS)
    parser.add_argument("--samples-per-task", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ruler_root = args.ruler_root.resolve()
    tokenizer_path = args.tokenizer_snapshot.resolve()
    if git_head(ruler_root) != RULER_REVISION:
        raise ValueError("RULER checkout revision does not match the frozen manifest.")
    if args.samples_per_task <= 0:
        raise ValueError("samples-per-task must be positive.")
    source = workspace_source()
    if source["dirty"]:
        raise RuntimeError("RULER generation requires a clean experiment source tree.")
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
            output_root=args.output_root,
            tokenizer=tokenizer,
            length=length,
            samples=args.samples_per_task,
            seed=args.seed,
            source=source,
            source_files=source_files,
        )


if __name__ == "__main__":
    main()
