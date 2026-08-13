from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import nano_deepseek_v4.compare_bundles as compare_module
from nano_deepseek_v4 import DeepSeekV4ForCausalLM
from nano_deepseek_v4.compare_bundles import compare_pretrained_bundles, main
from nano_deepseek_v4.train_text import tiny_text_config


@pytest.fixture
def byte_bundle_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(
        (b"sliding and compressed attention share paired byte windows.\n") * 40
    )
    torch.manual_seed(7)
    model = DeepSeekV4ForCausalLM(tiny_text_config())
    left = model.save_pretrained(tmp_path / "left")
    right = model.save_pretrained(tmp_path / "right")
    return corpus, left, right


def test_compare_identical_bundles_has_exactly_zero_paired_difference(
    byte_bundle_pair: tuple[Path, Path, Path],
):
    corpus, left, right = byte_bundle_pair

    report = compare_pretrained_bundles(
        text_file=corpus,
        left_bundle=left,
        right_bundle=right,
        left_label="first",
        right_label="second",
        batches=2,
        batch_size=1,
        context_length=8,
        validation_seed=19,
        device="cpu",
    )

    assert report.schema_version == 1
    assert report.evaluator_sha256 == hashlib.sha256(
        Path(compare_module.__file__).read_bytes()
    ).hexdigest()
    assert report.left.manifest_sha256 == report.right.manifest_sha256
    assert report.left.parameter_count == report.right.parameter_count == 246_590
    assert report.left.mtp_loss is not None
    assert report.right.mtp_loss is not None
    assert report.paired_left_minus_right_total.mean == 0.0
    assert report.paired_left_minus_right_total.descriptive_normal_approx_95pct_interval == (
        0.0,
        0.0,
    )
    assert report.paired_left_minus_right_causal_lm.mean == 0.0
    assert report.paired_left_minus_right_mtp is not None
    assert report.paired_left_minus_right_mtp.mean == 0.0
    assert not report.independent_window_assumption
    assert not report.training_seed_variance_measured


def test_compare_cli_writes_the_same_json_it_prints(
    byte_bundle_pair: tuple[Path, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    corpus, left, right = byte_bundle_pair
    output = tmp_path / "reports" / "comparison.json"

    return_code = main(
        [
            "--text-file",
            str(corpus),
            "--left-bundle",
            str(left),
            "--right-bundle",
            str(right),
            "--left-label",
            "first",
            "--right-label",
            "second",
            "--batches",
            "2",
            "--batch-size",
            "1",
            "--context-length",
            "8",
            "--validation-seed",
            "19",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--json",
        ]
    )
    printed = capsys.readouterr().out

    assert return_code == 0
    assert printed == output.read_text()
    payload = json.loads(printed)
    assert payload["left"]["label"] == "first"
    assert payload["right"]["label"] == "second"
    assert payload["paired_left_minus_right_total"]["mean"] == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batches": 1}, "batches"),
        ({"batch_size": 0}, "batch_size"),
        ({"context_length": 3}, "context_length"),
        ({"validation_seed": -1}, "validation_seed"),
        ({"left_label": "same", "right_label": "same"}, "must differ"),
    ],
)
def test_compare_rejects_invalid_arguments_before_opening_bundles(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
):
    arguments = {
        "text_file": tmp_path / "missing.txt",
        "left_bundle": tmp_path / "missing-left",
        "right_bundle": tmp_path / "missing-right",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        compare_pretrained_bundles(**arguments)  # type: ignore[arg-type]
