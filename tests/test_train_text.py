from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TypedDict

import pytest

import nano_deepseek_v4.train_text as train_module
from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM
from nano_deepseek_v4.train_text import (
    ByteTokenizer,
    main,
    mini_text_config,
    run_tiny_text_training,
    tiny_text_config,
)

_BASELINE_PATH = (
    Path(__file__).resolve().parents[1] / "references" / "tiny-text-training-baseline.json"
)
_SHAKESPEARE_BASELINE_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "tiny-shakespeare-training-baseline.json"
)
_ATTENTION_CONTROL_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "tiny-shakespeare-attention-control.json"
)
_ATTENTION_EVALUATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "tiny-shakespeare-attention-control-evaluation.json"
)
_ALL_SLIDING_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "tiny-shakespeare-all-sliding-control-config.json"
)


class _DeterministicTrainingArguments(TypedDict):
    text_file: Path
    steps: int
    context_length: int
    batch_size: int
    eval_batches: int
    learning_rate: float
    seed: int
    device: str
    max_new_tokens: int


class _SampledTrainingArguments(_DeterministicTrainingArguments):
    prompt: str
    temperature: float
    top_p: float


def _write_corpus(path: Path) -> Path:
    path.write_bytes(
        (
            b"sliding attention remembers nearby bytes; "
            b"compressed attention remembers distant bytes.\n"
        )
        * 80
    )
    return path


def test_byte_tokenizer_round_trips_every_byte():
    tokenizer = ByteTokenizer()
    payload = bytes(range(256))

    encoded = tokenizer.encode(payload, add_bos=True, add_eos=True)

    assert tokenizer.vocab_size == 259
    assert encoded[0].item() == tokenizer.bos_token_id
    assert encoded[-1].item() == tokenizer.eos_token_id
    assert tokenizer.decode_bytes(encoded) == payload


def test_tiny_text_config_exercises_every_attention_family():
    config = tiny_text_config()

    assert config.layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert config.mlp_layer_types == ["hash_moe", "moe", "moe"]
    assert config.num_nextn_predict_layers == 1
    assert config.vocab_size == ByteTokenizer().vocab_size


def test_mini_text_config_is_a_trainable_hybrid_corpus_model():
    config = mini_text_config()
    model = DeepSeekV4ForCausalLM(config)

    assert config.layer_types == [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert config.mlp_layer_types == ["hash_moe", "moe", "moe"]
    assert config.num_nextn_predict_layers == 1
    assert config.tie_word_embeddings
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert sum(parameter.numel() for parameter in model.parameters()) == 8_502_150


def test_tiny_text_training_saves_reproducible_bundle(tmp_path: Path):
    corpus = _write_corpus(tmp_path / "corpus.txt")
    bundle = tmp_path / "trained"

    result = run_tiny_text_training(
        text_file=corpus,
        steps=2,
        context_length=8,
        batch_size=1,
        eval_batches=1,
        learning_rate=1e-3,
        seed=5,
        device="cpu",
        max_new_tokens=1,
        save_directory=bundle,
    )
    loaded = DeepSeekV4ForCausalLM.from_pretrained(bundle)

    assert result.source == str(corpus)
    assert result.schema_version == 3
    assert len(result.corpus_sha256) == 64
    assert len(result.config_sha256) == 64
    assert len(result.implementation_sha256) == 64
    assert result.config_source == "tiny_text_config"
    assert result.python_version
    assert result.torch_version
    assert result.machine
    assert result.steps == 2
    assert result.eval_batches == 1
    assert result.learning_rate == 1e-3
    assert result.max_new_tokens == 1
    assert result.prompt_text
    assert result.generation_temperature is None
    assert result.generation_top_p == 1.0
    assert result.optimizer_names == ("Muon", "AdamW")
    assert result.trained_tokens == 16
    assert result.parameter_count == sum(parameter.numel() for parameter in loaded.parameters())
    assert result.trainable_parameter_count == result.parameter_count
    assert math.isfinite(result.initial_eval_loss)
    assert math.isfinite(result.final_eval_loss)
    assert math.isfinite(result.final_train_loss)
    assert result.tokens_per_second > 0
    assert result.save_directory == str(bundle)
    assert result.bundle_format_version == 2
    assert result.bundle_manifest_sha256 is not None
    assert len(result.bundle_manifest_sha256) == 64
    assert result.tokenizer_sha256 is not None
    assert len(result.tokenizer_sha256) == 64
    assert result.checkpoint_round_trip_match is True
    assert result.tokenizer_round_trip_match is True


def test_tiny_text_training_is_seed_deterministic(tmp_path: Path):
    corpus = _write_corpus(tmp_path / "deterministic.txt")
    arguments: _DeterministicTrainingArguments = {
        "text_file": corpus,
        "steps": 1,
        "context_length": 8,
        "batch_size": 1,
        "eval_batches": 1,
        "learning_rate": 1e-3,
        "seed": 17,
        "device": "cpu",
        "max_new_tokens": 0,
    }

    first = run_tiny_text_training(**arguments)
    second = run_tiny_text_training(**arguments)

    assert first.initial_eval_loss == second.initial_eval_loss
    assert first.final_eval_loss == second.final_eval_loss
    assert first.final_train_loss == second.final_train_loss
    assert first.sample_text == second.sample_text
    assert first.corpus_sha256 == second.corpus_sha256
    assert first.config_sha256 == second.config_sha256
    assert first.implementation_sha256 == second.implementation_sha256


def test_seeded_sampled_generation_is_deterministic(tmp_path: Path):
    corpus = _write_corpus(tmp_path / "sampled.txt")
    arguments: _SampledTrainingArguments = {
        "text_file": corpus,
        "steps": 1,
        "context_length": 8,
        "batch_size": 1,
        "eval_batches": 1,
        "learning_rate": 1e-3,
        "seed": 23,
        "device": "cpu",
        "max_new_tokens": 6,
        "prompt": "sliding",
        "temperature": 0.8,
        "top_p": 0.9,
    }

    first = run_tiny_text_training(**arguments)
    second = run_tiny_text_training(**arguments)

    assert first.sample_text == second.sample_text
    assert first.prompt_text == "sliding"
    assert first.generation_temperature == 0.8
    assert first.generation_top_p == 0.9


def test_builtin_training_baseline_improves_validation_loss(tmp_path: Path):
    baseline = json.loads(_BASELINE_PATH.read_text())
    expected = baseline["expected_result"]
    result = run_tiny_text_training(
        steps=20,
        context_length=32,
        batch_size=4,
        eval_batches=4,
        learning_rate=2e-3,
        seed=0,
        device="cpu",
        max_new_tokens=32,
        save_directory=tmp_path / "baseline-bundle",
    )

    assert result.loss_improved
    assert (
        result.initial_eval_loss - result.final_eval_loss
        > baseline["acceptance"]["minimum_validation_loss_reduction"]
    )
    assert result.corpus_sha256 == expected["corpus_sha256"]
    assert result.config_sha256 == expected["config_sha256"]
    historical_run = next(
        run
        for run in baseline["historical_runs"]
        if run["implementation_sha256"] == result.implementation_sha256
    )
    observations = historical_run["observations"]
    current_environment = {
        "device": result.device,
        "machine": result.machine,
        "python_version": result.python_version,
        "torch_version": result.torch_version,
    }
    if current_environment == historical_run["environment"]:
        assert result.initial_eval_loss == observations["initial_eval_loss"]
        assert result.final_eval_loss == observations["final_eval_loss"]
        assert result.final_train_loss == observations["final_train_loss"]
        assert result.sample_text == observations["sample_text"]
        assert result.bundle_manifest_sha256 == observations["bundle_manifest_sha256"]
        assert result.tokenizer_sha256 == observations["tokenizer_sha256"]
    assert result.schema_version == expected["result_schema_version"]
    assert result.bundle_format_version == expected["bundle_format_version"]
    assert result.tokenizer_round_trip_match is expected["tokenizer_round_trip_match"]
    assert result.eval_batches == baseline["recipe"]["eval_batches"]
    assert result.learning_rate == baseline["recipe"]["learning_rate"]
    assert result.max_new_tokens == baseline["recipe"]["max_new_tokens"]
    assert list(result.optimizer_names) == expected["optimizer_names"]
    assert result.parameter_count == expected["parameter_count"]
    assert result.trained_tokens == expected["trained_tokens"]
    assert (
        result.checkpoint_round_trip_match
        is expected["checkpoint_round_trip_match"]
    )


def test_checked_in_training_receipt_satisfies_its_acceptance_contract():
    baseline = json.loads(_BASELINE_PATH.read_text())

    assert baseline["schema_version"] == 3
    assert baseline["contract_id"] == "tiny-text-cpu-v1"
    for historical_run in baseline["historical_runs"]:
        observed = historical_run["observations"]
        assert (
            observed["initial_eval_loss"] - observed["final_eval_loss"]
            > baseline["acceptance"]["minimum_validation_loss_reduction"]
        )
        assert observed["checkpoint_round_trip_match"] is True
        assert observed["tokenizer_round_trip_match"] is True


def test_tiny_shakespeare_receipt_satisfies_its_acceptance_contract():
    baseline = json.loads(_SHAKESPEARE_BASELINE_PATH.read_text())
    observed = baseline["observed"]
    acceptance = baseline["acceptance"]
    canonical_config = json.dumps(
        mini_text_config().to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert baseline["schema_version"] == 2
    assert baseline["result_schema_version"] == 2
    assert baseline["corpus"]["sha256"] == (
        "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
    )
    assert baseline["experiment"]["config_sha256"] == hashlib.sha256(
        canonical_config
    ).hexdigest()
    assert len(baseline["experiment"]["implementation_sha256"]) == 64
    assert baseline["experiment"]["parameter_count"] == 8_502_150
    assert baseline["experiment"]["trained_tokens"] == (
        baseline["experiment"]["steps"]
        * baseline["experiment"]["batch_size"]
        * baseline["experiment"]["context_length"]
    )
    assert observed["initial_eval_loss"] - observed["final_eval_loss"] > (
        acceptance["minimum_validation_loss_reduction"]
    )
    assert observed["final_eval_loss"] < acceptance["maximum_final_eval_loss"]
    assert (
        observed["checkpoint_round_trip_match"]
        is acceptance["checkpoint_round_trip_match"]
    )
    assert len(observed["bundle_manifest_sha256"]) == 64
    assert [probe["prompt"] for probe in baseline["generation_probes"]] == [
        "\nJULIET:",
        "\nKING RICHARD III:",
        "\nFirst Citizen:",
    ]


def test_attention_control_receipt_is_bound_and_respects_its_claim_boundary():
    control = json.loads(_ATTENTION_CONTROL_PATH.read_text())
    evaluation_bytes = _ATTENTION_EVALUATION_PATH.read_bytes()
    evaluation = json.loads(evaluation_bytes)
    hybrid_receipt_bytes = _SHAKESPEARE_BASELINE_PATH.read_bytes()
    config_bytes = _ALL_SLIDING_CONFIG_PATH.read_bytes()
    evaluator_path = (
        Path(__file__).resolve().parents[1]
        / "nano_deepseek_v4"
        / "compare_bundles.py"
    )
    config = DeepSeekV4Config.from_json_file(_ALL_SLIDING_CONFIG_PATH)
    canonical_config = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert control["schema_version"] == 1
    assert control["status"] == "complete"
    assert control["artifacts"]["hybrid_receipt_sha256"] == hashlib.sha256(
        hybrid_receipt_bytes
    ).hexdigest()
    assert control["artifacts"]["all_sliding_config_file_sha256"] == hashlib.sha256(
        config_bytes
    ).hexdigest()
    assert control["artifacts"]["evaluation_report_sha256"] == hashlib.sha256(
        evaluation_bytes
    ).hexdigest()
    evaluator_sha256 = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    assert control["artifacts"]["evaluator_source_sha256"] == evaluator_sha256
    assert evaluation["evaluator_sha256"] == evaluator_sha256
    assert config.layer_types == ["sliding_attention"] * 3
    assert config.mtp_layer_types == ["sliding_attention"]
    assert control["runs"]["all_sliding"]["config_sha256"] == hashlib.sha256(
        canonical_config
    ).hexdigest()
    assert control["design"]["implementation_sha256"] == json.loads(
        _SHAKESPEARE_BASELINE_PATH.read_text()
    )["experiment"]["implementation_sha256"]

    acceptance = control["acceptance"]
    for run in control["runs"].values():
        assert run["initial_eval_loss"] - run["final_eval_loss"] > (
            acceptance["minimum_validation_loss_reduction"]
        )
        assert run["final_eval_loss"] < acceptance["maximum_final_eval_loss"]
        assert (
            run["checkpoint_round_trip_match"]
            is acceptance["checkpoint_round_trip_match"]
        )
        assert len(run["bundle_manifest_sha256"]) == 64

    expected_throughput_gain = (
        control["runs"]["all_sliding"]["tokens_per_second"]
        / control["runs"]["hybrid"]["tokens_per_second"]
        - 1
    ) * 100
    assert control["derived_comparison"][
        "all_sliding_throughput_percent_higher"
    ] == pytest.approx(expected_throughput_gain)
    evaluation_metric_names = {
        "total": "paired_left_minus_right_total",
        "causal_lm": "paired_left_minus_right_causal_lm",
        "mtp": "paired_left_minus_right_mtp",
    }
    for metric, evaluation_name in evaluation_metric_names.items():
        assert control["paired_evaluation"]["paired_hybrid_minus_all_sliding"][
            metric
        ]["mean"] == evaluation[evaluation_name]["mean"]
        lower, upper = control["paired_evaluation"][
            "paired_hybrid_minus_all_sliding"
        ][metric]["descriptive_normal_approx_95pct_interval"]
        assert lower <= 0 <= upper

    assert evaluation["left"]["manifest_sha256"] == control["runs"]["hybrid"][
        "bundle_manifest_sha256"
    ]
    assert evaluation["right"]["manifest_sha256"] == control["runs"][
        "all_sliding"
    ]["bundle_manifest_sha256"]

    assert any("single-training-seed" in item for item in control["claim_boundary"])
    assert any("descriptive" in item for item in control["claim_boundary"])


def test_tiny_text_cli_emits_machine_readable_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    corpus = _write_corpus(tmp_path / "cli.txt")
    output = tmp_path / "reports" / "training.json"

    return_code = main(
        [
            "--text-file",
            str(corpus),
            "--steps",
            "1",
            "--context-length",
            "8",
            "--batch-size",
            "1",
            "--eval-batches",
            "1",
            "--max-new-tokens",
            "0",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--json",
        ]
    )
    printed = capsys.readouterr().out
    payload = json.loads(printed)

    assert return_code == 0
    assert printed == output.read_text(encoding="utf-8")
    assert payload["source"] == str(corpus)
    assert payload["schema_version"] == 3
    assert payload["steps"] == 1
    assert payload["trained_tokens"] == 8


def test_tiny_text_cli_accepts_a_native_config_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    corpus = _write_corpus(tmp_path / "custom-config-corpus.txt")
    config_path = tmp_path / "config.json"
    tiny_text_config().to_json_file(config_path)

    return_code = main(
        [
            "--text-file",
            str(corpus),
            "--config",
            str(config_path),
            "--steps",
            "1",
            "--context-length",
            "8",
            "--batch-size",
            "1",
            "--eval-batches",
            "1",
            "--max-new-tokens",
            "0",
            "--device",
            "cpu",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload["config_source"] == str(config_path)
    assert len(payload["config_sha256"]) == 64
    assert payload["parameter_count"] == 246_590


def test_tiny_text_training_rejects_too_short_corpus(tmp_path: Path):
    corpus = tmp_path / "short.txt"
    corpus.write_text("too short")

    with pytest.raises(ValueError, match="at least 18 bytes"):
        run_tiny_text_training(
            text_file=corpus,
            steps=1,
            context_length=8,
            batch_size=1,
            eval_batches=1,
            device="cpu",
            max_new_tokens=0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vocab_size": 260}, "vocab_size must match"),
        ({"pad_token_id": 3}, "pad_token_id must be 0"),
    ],
)
def test_tiny_text_training_rejects_incompatible_tokenizer_config(overrides, message):
    values = tiny_text_config().to_dict()
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        run_tiny_text_training(
            config=DeepSeekV4Config.from_dict(values),
            steps=1,
            context_length=8,
            batch_size=1,
            eval_batches=1,
            device="cpu",
            max_new_tokens=0,
        )


def test_tiny_text_training_rejects_top_p_without_sampling():
    with pytest.raises(ValueError, match="top_p requires temperature"):
        run_tiny_text_training(
            steps=1,
            context_length=8,
            batch_size=1,
            eval_batches=1,
            device="cpu",
            max_new_tokens=0,
            top_p=0.9,
        )


def test_tiny_text_training_rejects_prompt_overflow_before_training(
    monkeypatch: pytest.MonkeyPatch,
):
    values = tiny_text_config().to_dict()
    values["max_position_embeddings"] = 16
    config = DeepSeekV4Config.from_dict(values)
    monkeypatch.setattr(
        train_module,
        "train_step",
        lambda *args, **kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(ValueError, match="prompt tokens plus max_new_tokens"):
        run_tiny_text_training(
            config=config,
            steps=1,
            context_length=8,
            batch_size=1,
            eval_batches=1,
            device="cpu",
            max_new_tokens=1,
            prompt="x" * 16,
        )


def test_tiny_text_training_rejects_occupied_save_directory_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    destination = tmp_path / "occupied"
    destination.mkdir()
    marker = destination / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "train_step",
        lambda *args, **kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(ValueError, match="save_directory must be absent or empty"):
        run_tiny_text_training(
            steps=1,
            context_length=8,
            batch_size=1,
            eval_batches=1,
            device="cpu",
            max_new_tokens=0,
            save_directory=destination,
        )

    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_tiny_text_training_reserves_seeded_substreams_before_training(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        train_module,
        "train_step",
        lambda *args, **kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(ValueError, match=r"seed must be an integer in \[0,"):
        run_tiny_text_training(
            steps=1,
            context_length=8,
            batch_size=1,
            eval_batches=1,
            device="cpu",
            max_new_tokens=0,
            seed=2**64 - 3,
        )


def test_tiny_text_training_accepts_largest_seed_with_all_substreams():
    result = run_tiny_text_training(
        steps=1,
        context_length=8,
        batch_size=1,
        eval_batches=1,
        device="cpu",
        max_new_tokens=0,
        seed=2**64 - 4,
    )

    assert result.seed == 2**64 - 4


@pytest.mark.parametrize(
    ("save_suffix", "output_suffix"),
    [
        ("artifact", "artifact"),
        ("artifact", "artifact/receipt.json"),
        ("artifact/bundle", "artifact"),
    ],
)
def test_tiny_text_cli_rejects_overlapping_output_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    save_suffix: str,
    output_suffix: str,
):
    monkeypatch.setattr(
        train_module,
        "run_tiny_text_training",
        lambda **kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--save-directory",
                str(tmp_path / save_suffix),
                "--output",
                str(tmp_path / output_suffix),
            ]
        )

    assert exc_info.value.code == 2
    assert "--output and --save-directory must not overlap" in capsys.readouterr().err


def test_tiny_text_cli_reports_runtime_failure_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    def fail_training(**kwargs):
        raise RuntimeError("training produced a non-finite loss")

    monkeypatch.setattr(train_module, "run_tiny_text_training", fail_training)

    with pytest.raises(SystemExit) as exc_info:
        main([])

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 2
    assert "training produced a non-finite loss" in stderr
    assert "Traceback" not in stderr
