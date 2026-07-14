from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from p3_natural_workloads import load_official_scbench_module, sha256
from p3_scbench_metrics import official_ground_truth, score_turn

REPO_TASKS = frozenset({"scbench_repoqa", "scbench_repoqa_and_kv"})
REPO_THRESHOLD = 0.8
ROUGE_REVISION = "ea7c4bf30945a2a8e31f2b1b3bdba6cd617eebe2"
ROUGE_SCRIPT_SHA256 = "805b71b855be6dc270ce2366639a4e264c2a1fb93a2c8a50f9fb9ef5a2b9c21a"


def load_repoqa_module(source_root: Path, expected_sha256: str) -> Any:
    path = source_root / "scbench" / "repo_qa_utils.py"
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError("Pinned SCBench repo_qa_utils.py SHA-256 drifted.")
    spec = importlib.util.spec_from_file_location("adaptive_v4_scbench_repoqa", path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not construct the pinned RepoQA module loader.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_rouge_lsum() -> Any:
    import evaluate
    from huggingface_hub import hf_hub_download

    script = Path(
        hf_hub_download(
            repo_id="evaluate-metric/rouge",
            filename="rouge.py",
            repo_type="space",
            revision=ROUGE_REVISION,
        )
    )
    if sha256(script) != ROUGE_SCRIPT_SHA256:
        raise ValueError("Pinned ROUGE metric script SHA-256 drifted.")
    return evaluate.load(str(script))


def build_repo_needles(
    rows_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    needles: dict[str, dict[str, list[dict[str, str]]]] = {}
    for task in REPO_TASKS:
        task_needles: dict[str, list[dict[str, str]]] = {}
        for row in rows_by_task.get(task, []):
            repo = row.get("repo")
            if not isinstance(repo, str) or not repo:
                raise ValueError(f"{task} row has no repository identity.")
            for turn in row.get("multi_turns", []):
                if task == "scbench_repoqa_and_kv" and turn.get("task") != "scbench_repoqa":
                    continue
                name, answer = turn.get("name"), turn.get("answer")
                if not isinstance(name, str) or not name or not isinstance(answer, str):
                    raise ValueError(f"{task} RepoQA turn is missing a function needle.")
                task_needles.setdefault(repo, []).append({"name": name, "needle": answer})
        for repo, values in task_needles.items():
            unique = {(value["name"], value["needle"]) for value in values}
            if len(unique) != len(values):
                raise ValueError(f"{task} contains duplicate needles for {repo}.")
        needles[task] = task_needles
    return needles


class OfficialSCBenchScorer:
    def __init__(
        self,
        *,
        rows_by_task: dict[str, list[dict[str, Any]]],
        repo_module: Any,
        rouge_metric: Any,
    ) -> None:
        self.repo_module = repo_module
        self.rouge_metric = rouge_metric
        self.repo_needles = build_repo_needles(rows_by_task)

    def _rouge_lsum(self, prediction: str, reference: str) -> float:
        result = self.rouge_metric.compute(
            predictions=[prediction],
            references=[reference],
            use_aggregator=False,
        )
        scores = result["rougeLsum"]
        if not isinstance(scores, list) or len(scores) != 1:
            raise ValueError("Pinned SCBench ROUGE-Lsum returned an invalid result.")
        return float(scores[0])

    def score(
        self,
        *,
        task: str,
        row: dict[str, Any],
        turn: dict[str, Any],
        prediction: str,
        subtask: str | None,
    ) -> tuple[float, dict[str, Any]]:
        effective = subtask or task
        if effective == "scbench_repoqa":
            repo, language = row.get("repo"), row.get("lang")
            name, answer = turn.get("name"), turn.get("answer")
            if (
                not isinstance(repo, str)
                or task not in self.repo_needles
                or repo not in self.repo_needles[task]
                or not isinstance(language, str)
                or not isinstance(name, str)
                or not isinstance(answer, str)
            ):
                raise ValueError("RepoQA record lacks repository-aware scoring fields.")
            verdict, best_target, similarity = self.repo_module.needle_evaluator(
                prediction,
                {"func_name": name, "ground_truth": answer},
                self.repo_needles[task][repo],
                language,
                False,
            )
            correct = (
                verdict == self.repo_module.Result.BEST_MATCH
                and float(similarity) >= REPO_THRESHOLD
            )
            return float(correct), {
                "metric": "repoqa-best-match-pass@1",
                "threshold": REPO_THRESHOLD,
                "best_target": best_target,
                "best_similarity": float(similarity),
            }

        ground_truth = official_ground_truth(task, turn)
        score = score_turn(
            task=task,
            prediction=prediction,
            ground_truth=ground_truth,
            subtask=subtask,
            rouge_lsum=self._rouge_lsum,
        )
        return score, {"metric": effective}


def load_official_components(
    source_root: Path, file_digests: dict[str, str]
) -> tuple[Any, Any, Any]:
    prompt_module = load_official_scbench_module(source_root, file_digests["scbench/eval_utils.py"])
    repo_module = load_repoqa_module(source_root, file_digests["scbench/repo_qa_utils.py"])
    rouge_metric = load_rouge_lsum()
    return prompt_module, repo_module, rouge_metric
