from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from run_p3_natural_ruler import (  # noqa: E402
    EXPECTED_EXAMPLES,
    expected_example_ids,
    load_dataset_contracts,
    rendered_input,
    token_digest,
)


class FakePipeline:
    def preprocess(self, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["max_context_length"] == 262144
        return {
            "context_ids": torch.tensor([[1, 2, 3]]),
            "questions_ids": [torch.tensor([[4, 5]])],
        }


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

    assert result["exact_input_tokens"] == 5
    assert result["raw_prompt_sha256"] == token_digest(
        torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5]])
    )
    assert result["raw_prompt_sha256"] == hashlib_digest(
        {"context_ids": [1, 2, 3], "question_ids": [4, 5]}
    )


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
                    "ruler": {"revision": "38da79d79519ef87aa46ae804f838e1eab7f86d7"},
                    "tokenizer": {
                        "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554"
                    },
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


def hashlib_digest(payload: dict) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
