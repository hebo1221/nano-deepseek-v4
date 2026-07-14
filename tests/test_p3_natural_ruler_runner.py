from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_natural_ruler import (  # noqa: E402
    EXPECTED_EXAMPLES,
    _existing_records,
    compatibility_arm_config,
    expected_example_ids,
    load_dataset_contracts,
    rendered_input,
    select_score_compatible_baseline,
    token_digest,
)


class FakeTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        assert kwargs["enable_thinking"] is False
        return f"<user>{messages[0]['content']}</user><assistant>"

    def encode(self, text: str, **_kwargs: object) -> torch.Tensor:
        values = [
            sum(ord(char) for char in text[index : index + 2]) for index in range(0, len(text), 2)
        ]
        return torch.tensor(values, dtype=torch.long).unsqueeze(0)


class FakePipeline:
    tokenizer = FakeTokenizer()


def test_natural_ruler_identity_grid_closes_all_lengths_tasks_and_rows() -> None:
    identifiers = expected_example_ids()

    assert len(identifiers) == EXPECTED_EXAMPLES == 32_500
    assert len(set(identifiers)) == EXPECTED_EXAMPLES
    assert identifiers[0].startswith("8192:niah_single_1:0")
    assert identifiers[-1].startswith("131072:qa_2:499")


def test_natural_ruler_rendering_records_exact_token_identity() -> None:
    result = rendered_input(
        FakePipeline(),
        context="context",
        question="question",
        answer_prefix="answer",
        maximum_context=262144,
    )

    full = "<user>contextquestion</user><assistant>answer"
    full_ids = FakePipeline.tokenizer.encode(full, add_special_tokens=False)
    assert result["exact_input_tokens"] == full_ids.shape[1]
    assert torch.equal(torch.cat((result["context_ids"], result["question_ids"]), dim=1), full_ids)
    assert result["input_token_ids_sha256"] == token_digest(
        result["context_ids"], result["question_ids"]
    )
    assert result["raw_prompt_sha256"] == hashlib.sha256(full.encode()).hexdigest()


def test_natural_ruler_dataset_contracts_bind_every_length(tmp_path: Path) -> None:
    manifest_digest = "a" * 64
    generator_digest = "b" * 64
    for length in (8192, 16384, 32768, 65536, 131072):
        path = tmp_path / str(length) / "dataset-manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "experiment_id": "p3-natural-ruler-qwen3-4b-dataset-v1",
                    "source": {"dirty": False, "implementation_sha256": generator_digest},
                    "natural_suite_manifest": {"sha256": manifest_digest},
                    "ruler": {
                        "revision": "38da79d79519ef87aa46ae804f838e1eab7f86d7",
                        "clean_tracked_tree": True,
                    },
                    "tokenizer": {"model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"},
                    "generation": {
                        "length_tokens": length,
                        "samples_per_task": 500,
                        "tasks": [
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
                        ],
                        "total_rows": 6500,
                    },
                }
            )
        )

    contracts, digest_set = load_dataset_contracts(
        tmp_path,
        natural_manifest_digest=manifest_digest,
        generator_digest=generator_digest,
    )

    assert set(contracts) == {8192, 16384, 32768, 65536, 131072}
    assert len(digest_set) == 64


def test_natural_ruler_progress_recovers_empty_crash_window(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    partial = tmp_path / "records.partial.jsonl"
    identity = {"digest": "a" * 64}

    assert _existing_records(progress, partial, identity) == []
    assert progress.is_file() and partial.is_file()
    partial.unlink()
    assert _existing_records(progress, partial, identity) == []
    progress.unlink()
    assert _existing_records(progress, partial, identity) == []


def test_adaptive_quota_cohort_selects_best_direct_scorer_without_natural_outcomes() -> None:
    selection = {
        "candidates": [
            {
                "arm": arm,
                "compression_ratio": 0.5,
                "row_weighted_mean_accuracy": score,
            }
            for arm, score in (
                ("streaming_llm", 0.70),
                ("snapkv", 0.75),
                ("pyramidkv", 0.99),
                ("critical_expected_attention", 0.80),
                ("adakv_snapkv", 0.99),
                ("expected_attention", 0.98),
            )
        ]
    }

    assert select_score_compatible_baseline(selection)["arm"] == "critical_expected_attention"
    fixed = compatibility_arm_config("fixed+pins", selection, "a" * 64)
    adaptive = compatibility_arm_config("natural-adaptive-quota+pins", selection, "a" * 64)
    assert fixed["press_name"] == adaptive["press_name"] == "critical_expected_attention"
    assert fixed["compression_ratio"] == adaptive["compression_ratio"] == 0.5
    assert fixed["max_adjustment_fraction"] == 0.0
    assert adaptive["max_adjustment_fraction"] == 0.25
    assert fixed["protected_prefix_token_span"] == {"start": 0, "end": 4}


def test_natural_ruler_is_sequence_gated_before_model_or_dataset_io(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    p2 = tmp_path / "p2.json"
    p2.write_text(
        json.dumps({"completed_shards": 12, "frozen_design": {"total_expected_shards": 4500}})
    )
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "research/adaptive_v4_memory/scripts/run_p3_natural_ruler.py"),
            "--kvpress-root",
            str(tmp_path / "missing-kvpress"),
            "--model-snapshot",
            str(tmp_path / "missing-model"),
            "--p2-matrix",
            str(p2),
            "--causal-gate",
            str(tmp_path / "missing-causal.json"),
            "--output-root",
            str(output),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "12/4500 shards" in completed.stderr
    assert not output.exists()
