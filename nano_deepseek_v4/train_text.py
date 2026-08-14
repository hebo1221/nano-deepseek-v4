from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_deepseek_v4_pretrained_tokenizer
from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM
from .tokenizer import ByteTokenizer
from .training import build_deepseek_v4_optimizers, train_step

_BUILTIN_CORPUS = (
    "DeepSeek V4 combines sliding attention with compressed sparse attention.\n"
    "Heavily compressed attention preserves a compact long-range memory.\n"
    "Hash routing bootstraps experts before learned routing takes over.\n"
    "Multi-token prediction teaches each hidden state to look further ahead.\n"
    "A readable implementation should make every tensor transformation inspectable.\n"
) * 48

_IMPLEMENTATION_FILES = (
    "checkpoint.py",
    "config.py",
    "modeling.py",
    "optim.py",
    "tokenizer.py",
    "training.py",
    "train_text.py",
)
_MAX_TRAINING_SEED = 2**64 - 4


@dataclass(frozen=True)
class TinyTextTrainingResult:
    schema_version: int
    source: str
    corpus_sha256: str
    config_source: str
    config_sha256: str
    implementation_sha256: str
    python_version: str
    torch_version: str
    machine: str
    device: str
    seed: int
    steps: int
    context_length: int
    batch_size: int
    eval_batches: int
    learning_rate: float
    max_new_tokens: int
    prompt_text: str
    generation_temperature: float | None
    generation_top_p: float
    optimizer_names: tuple[str, ...]
    parameter_count: int
    trainable_parameter_count: int
    initial_eval_loss: float
    final_eval_loss: float
    final_train_loss: float
    loss_improved: bool
    elapsed_seconds: float
    trained_tokens: int
    tokens_per_second: float
    sample_text: str
    save_directory: str | None
    bundle_format_version: int | None
    bundle_manifest_sha256: str | None
    tokenizer_sha256: str | None
    checkpoint_round_trip_match: bool | None
    tokenizer_round_trip_match: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tiny_text_config() -> DeepSeekV4Config:
    """Return the 247K-parameter hybrid configuration used by the CLI."""

    return DeepSeekV4Config(
        vocab_size=259,
        hidden_size=48,
        moe_intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=12,
        q_lora_rank=24,
        num_experts_per_tok=2,
        n_routed_experts=4,
        n_shared_experts=1,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        num_hash_layers=1,
        hc_mult=2,
        sliding_window=16,
        o_groups=2,
        o_lora_rank=12,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=4,
        num_nextn_predict_layers=1,
        partial_rotary_factor=0.5,
        compress_rates={
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def mini_text_config() -> DeepSeekV4Config:
    """Return an 8.50M-parameter hybrid config for real small-corpus training."""

    return DeepSeekV4Config(
        vocab_size=259,
        hidden_size=192,
        moe_intermediate_size=384,
        num_hidden_layers=3,
        num_attention_heads=6,
        head_dim=32,
        q_lora_rank=96,
        num_experts_per_tok=2,
        n_routed_experts=8,
        n_shared_experts=1,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        num_hash_layers=1,
        hc_mult=2,
        sliding_window=64,
        o_groups=3,
        o_lora_rank=48,
        index_n_heads=6,
        index_head_dim=16,
        index_topk=8,
        num_nextn_predict_layers=1,
        partial_rotary_factor=0.5,
        compress_rates={
            "compressed_sparse_attention": 8,
            "heavily_compressed_attention": 32,
        },
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str) and device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    if resolved.type == "meta":
        raise ValueError("the training device must materialize tensors.")
    return resolved


def _validate_training_arguments(
    *,
    steps: int,
    context_length: int,
    batch_size: int,
    eval_batches: int,
    learning_rate: float,
    seed: int,
    max_new_tokens: int,
    temperature: float | None,
    top_p: float,
) -> None:
    positive_integers = {
        "steps": steps,
        "batch_size": batch_size,
        "eval_batches": eval_batches,
    }
    for name, value in positive_integers.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 4
    ):
        raise ValueError("context_length must be an integer of at least 4.")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > _MAX_TRAINING_SEED
    ):
        raise ValueError(f"seed must be an integer in [0, {_MAX_TRAINING_SEED}].")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError("max_new_tokens must be a non-negative integer.")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be a positive finite number.")
    if temperature is not None and (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError("temperature must be None or a positive finite number.")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0 < top_p <= 1
    ):
        raise ValueError("top_p must be in (0, 1].")
    if temperature is None and top_p != 1.0:
        raise ValueError("top_p requires temperature sampling.")


def _validate_training_config(
    config: DeepSeekV4Config,
    tokenizer: ByteTokenizer,
    context_length: int,
) -> None:
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "training config vocab_size must match the byte tokenizer: "
            f"expected {tokenizer.vocab_size}, got {config.vocab_size}."
        )
    expected_special_ids = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    for name, expected in expected_special_ids.items():
        if getattr(config, name) != expected:
            raise ValueError(f"training config {name} must be {expected}.")
    if context_length > config.max_position_embeddings:
        raise ValueError(
            f"context_length={context_length} exceeds config max_position_embeddings="
            f"{config.max_position_embeddings}."
        )


def _validate_save_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("save_directory must not be a symbolic link.")
    if not path.exists():
        return
    if not path.is_dir():
        raise ValueError("save_directory must be a directory path.")
    if next(path.iterdir(), None) is not None:
        raise ValueError("save_directory must be absent or empty.")


def _validate_output_location(
    save_directory: Path | None,
    output: Path | None,
) -> None:
    if save_directory is None or output is None:
        return
    destination = save_directory.resolve(strict=False)
    receipt = output.resolve(strict=False)
    if (
        receipt == destination
        or destination in receipt.parents
        or receipt in destination.parents
    ):
        raise ValueError("--output and --save-directory must not overlap.")


def _prepare_prompt(
    tokenizer: ByteTokenizer,
    prompt: str | bytes | None,
    *,
    validation_tokens: torch.Tensor,
    context_length: int,
    max_new_tokens: int,
    max_position_embeddings: int,
) -> tuple[torch.Tensor, str]:
    if prompt is None:
        prompt_tokens = validation_tokens[: min(16, context_length)]
    else:
        if not isinstance(prompt, (str, bytes)):
            raise TypeError("prompt must be text, bytes, or None.")
        prompt_tokens = tokenizer.encode(prompt)
        if prompt_tokens.numel() == 0:
            raise ValueError("prompt must not be empty.")
    if prompt_tokens.numel() + max_new_tokens > max_position_embeddings:
        raise ValueError(
            "prompt tokens plus max_new_tokens exceed config max_position_embeddings: "
            f"{prompt_tokens.numel()} + {max_new_tokens} > {max_position_embeddings}."
        )
    return prompt_tokens, tokenizer.decode(prompt_tokens)


def _load_token_stream(
    tokenizer: ByteTokenizer,
    text_file: str | Path | None,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    if text_file is None:
        payload = _BUILTIN_CORPUS.encode("utf-8")
        source = "built-in architecture corpus"
    else:
        path = Path(text_file)
        payload = path.read_bytes()
        source = str(path)
    tokens = tokenizer.encode(payload)
    minimum_tokens = 2 * (context_length + 1)
    if tokens.numel() < minimum_tokens:
        raise ValueError(
            f"training text must contain at least {minimum_tokens} bytes for "
            f"context_length={context_length}; got {tokens.numel()}."
        )

    split = int(tokens.numel() * 0.9)
    split = max(context_length + 1, split)
    split = min(tokens.numel() - context_length - 1, split)
    return tokens[:split], tokens[split:], source, hashlib.sha256(payload).hexdigest()


def _config_sha256(config: DeepSeekV4Config) -> str:
    canonical = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _implementation_sha256() -> str:
    """Fingerprint the local source files that define this experiment."""

    package_directory = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in _IMPLEMENTATION_FILES:
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update((package_directory / filename).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json_receipt(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sample_batch(
    stream: torch.Tensor,
    *,
    context_length: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    start_count = stream.numel() - context_length + 1
    starts = torch.randint(0, start_count, (batch_size,), generator=generator)
    rows = [stream[int(start) : int(start) + context_length] for start in starts]
    return torch.stack(rows).to(device)


@torch.no_grad()
def _evaluate_loss(
    model: DeepSeekV4ForCausalLM,
    batches: list[torch.Tensor],
) -> float:
    was_training = model.training
    model.eval()
    try:
        losses: list[float] = []
        for batch in batches:
            output = model(batch, labels=batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("evaluation produced a non-finite loss.")
            losses.append(float(output.loss))
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def run_tiny_text_training(
    *,
    text_file: str | Path | None = None,
    config: DeepSeekV4Config | None = None,
    config_source: str | None = None,
    steps: int = 20,
    context_length: int = 32,
    batch_size: int = 4,
    eval_batches: int = 4,
    learning_rate: float = 2e-3,
    seed: int = 0,
    device: str | torch.device = "auto",
    max_new_tokens: int = 32,
    prompt: str | bytes | None = None,
    temperature: float | None = None,
    top_p: float = 1.0,
    save_directory: str | Path | None = None,
) -> TinyTextTrainingResult:
    """Train every major architecture path on bytes and return reproducible metrics."""

    _validate_training_arguments(
        steps=steps,
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=eval_batches,
        learning_rate=learning_rate,
        seed=seed,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if save_directory is not None:
        _validate_save_directory(Path(save_directory))
    target_device = _resolve_device(device)
    tokenizer = ByteTokenizer()
    if config is None:
        config = tiny_text_config()
        resolved_config_source = config_source or "tiny_text_config"
    else:
        if not isinstance(config, DeepSeekV4Config):
            raise TypeError("config must be a DeepSeekV4Config or None.")
        resolved_config_source = config_source or "caller-provided config"
    _validate_training_config(config, tokenizer, context_length)
    train_tokens, validation_tokens, source, corpus_sha256 = _load_token_stream(
        tokenizer,
        text_file,
        context_length,
    )
    prompt_tokens, prompt_text = _prepare_prompt(
        tokenizer,
        prompt,
        validation_tokens=validation_tokens,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        max_position_embeddings=config.max_position_embeddings,
    )

    torch.manual_seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config_sha256 = _config_sha256(config)
    model = DeepSeekV4ForCausalLM(config).to(target_device)
    optimizers = build_deepseek_v4_optimizers(
        model,
        lr=float(learning_rate),
        weight_decay=0.0,
    )

    validation_generator = torch.Generator().manual_seed(seed + 1)
    validation_batches = [
        _sample_batch(
            validation_tokens,
            context_length=context_length,
            batch_size=batch_size,
            generator=validation_generator,
            device=target_device,
        )
        for _ in range(eval_batches)
    ]
    initial_eval_loss = _evaluate_loss(model, validation_batches)

    training_generator = torch.Generator().manual_seed(seed + 2)
    final_train_loss = float("nan")
    start_time = time.perf_counter()
    for _ in range(steps):
        batch = _sample_batch(
            train_tokens,
            context_length=context_length,
            batch_size=batch_size,
            generator=training_generator,
            device=target_device,
        )
        loss = train_step(model, batch, optimizers)
        if not torch.isfinite(loss):
            raise RuntimeError("training produced a non-finite loss.")
        final_train_loss = float(loss)
    elapsed_seconds = time.perf_counter() - start_time
    final_eval_loss = _evaluate_loss(model, validation_batches)

    prompt_batch = prompt_tokens.unsqueeze(0).to(target_device)
    generation_generator = torch.Generator(device=target_device).manual_seed(seed + 3)
    model.eval()
    generated = model.generate(
        prompt_batch,
        max_new_tokens=max_new_tokens,
        do_sample=temperature is not None,
        temperature=1.0 if temperature is None else float(temperature),
        top_p=float(top_p),
        generator=generation_generator,
        stop_on_eos=False,
    )
    sample_text = tokenizer.decode(generated[0])

    saved_path: str | None = None
    bundle_format_version: int | None = None
    bundle_manifest_sha256: str | None = None
    tokenizer_sha256: str | None = None
    round_trip_match: bool | None = None
    tokenizer_round_trip_match: bool | None = None
    if save_directory is not None:
        saved = model.save_pretrained(save_directory, tokenizer=tokenizer)
        restored = DeepSeekV4ForCausalLM.from_pretrained(saved, device=target_device)
        restored_tokenizer = load_deepseek_v4_pretrained_tokenizer(saved)
        reference_batch = validation_batches[0]
        with torch.no_grad():
            expected_logits = model(reference_batch).logits
            actual_logits = restored(reference_batch).logits
        round_trip_match = torch.equal(actual_logits, expected_logits)
        if not round_trip_match:
            raise RuntimeError("saved model bundle did not reproduce the trained logits.")
        saved_path = str(saved)
        manifest_path = saved / "nano_deepseek_v4.json"
        bundle_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle_format_version = int(manifest["format_version"])
        tokenizer_filename = manifest["tokenizer_file"]
        tokenizer_sha256 = manifest["sha256"][tokenizer_filename]
        tokenizer_round_trip_match = restored_tokenizer == tokenizer
        if not tokenizer_round_trip_match:
            raise RuntimeError("saved tokenizer sidecar did not round-trip exactly.")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trained_tokens = steps * batch_size * context_length
    return TinyTextTrainingResult(
        schema_version=3,
        source=source,
        corpus_sha256=corpus_sha256,
        config_source=resolved_config_source,
        config_sha256=config_sha256,
        implementation_sha256=_implementation_sha256(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        machine=platform.machine(),
        device=str(target_device),
        seed=seed,
        steps=steps,
        context_length=context_length,
        batch_size=batch_size,
        eval_batches=eval_batches,
        learning_rate=float(learning_rate),
        max_new_tokens=max_new_tokens,
        prompt_text=prompt_text,
        generation_temperature=(
            None if temperature is None else float(temperature)
        ),
        generation_top_p=float(top_p),
        optimizer_names=(type(optimizers.muon).__name__, type(optimizers.adamw).__name__),
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_parameter_count,
        initial_eval_loss=initial_eval_loss,
        final_eval_loss=final_eval_loss,
        final_train_loss=final_train_loss,
        loss_improved=final_eval_loss < initial_eval_loss,
        elapsed_seconds=elapsed_seconds,
        trained_tokens=trained_tokens,
        tokens_per_second=trained_tokens / max(elapsed_seconds, 1e-12),
        sample_text=sample_text,
        save_directory=saved_path,
        bundle_format_version=bundle_format_version,
        bundle_manifest_sha256=bundle_manifest_sha256,
        tokenizer_sha256=tokenizer_sha256,
        checkpoint_round_trip_match=round_trip_match,
        tokenizer_round_trip_match=tokenizer_round_trip_match,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tiny hybrid DeepSeek-V4 model on raw UTF-8 or byte text.",
    )
    parser.add_argument("--text-file", type=Path, help="Text file; omit for the built-in corpus.")
    model = parser.add_mutually_exclusive_group()
    model.add_argument(
        "--model-preset",
        choices=("tiny", "mini"),
        help="tiny is the fast integration smoke test; mini is an 8.50M corpus model.",
    )
    model.add_argument("--config", type=Path, help="Native DeepSeekV4Config JSON.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", help="Generation prompt; defaults to held-out corpus bytes.")
    parser.add_argument(
        "--temperature",
        type=float,
        help="Enable seeded sampling at this temperature; omit for greedy generation.",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--save-directory", type=Path)
    parser.add_argument("--output", type=Path, help="Atomically write the full JSON receipt.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable metrics.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_output_location(args.save_directory, args.output)
        if args.config is not None:
            config = DeepSeekV4Config.from_json_file(args.config)
            config_source = str(args.config)
        elif args.model_preset == "mini":
            config = mini_text_config()
            config_source = "mini_text_config"
        else:
            config = tiny_text_config()
            config_source = "tiny_text_config"
        result = run_tiny_text_training(
            text_file=args.text_file,
            config=config,
            config_source=config_source,
            steps=args.steps,
            context_length=args.context_length,
            batch_size=args.batch_size,
            eval_batches=args.eval_batches,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            prompt=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            save_directory=args.save_directory,
        )
        payload = json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        if args.output is not None:
            _write_json_receipt(args.output, payload)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(payload, end="")
    else:
        print("nano-deepseek-v4 tiny-text training")
        print(f"source: {result.source}")
        print(f"config: {result.config_source} ({result.parameter_count:,} parameters)")
        print(f"device: {result.device}")
        print(f"validation loss: {result.initial_eval_loss:.4f} -> {result.final_eval_loss:.4f}")
        print(
            f"trained: {result.trained_tokens:,} tokens in "
            f"{result.elapsed_seconds:.2f}s ({result.tokens_per_second:,.0f} tok/s)"
        )
        if result.save_directory is not None:
            print(f"bundle: {result.save_directory}")
            print(f"bundle format: v{result.bundle_format_version}")
            print(f"bundle manifest sha256: {result.bundle_manifest_sha256}")
            print(f"tokenizer sha256: {result.tokenizer_sha256}")
            print(f"bundle round-trip: {result.checkpoint_round_trip_match}")
            print(f"tokenizer round-trip: {result.tokenizer_round_trip_match}")
        if result.generation_temperature is not None:
            print(
                f"sampling: temperature={result.generation_temperature:g}, "
                f"top_p={result.generation_top_p:g}"
            )
        print(f"sample: {result.sample_text!r}")
        if args.output is not None:
            print(f"receipt: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
