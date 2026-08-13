from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .checkpoint import PretrainedBundleReport, verify_deepseek_v4_pretrained_bundle
from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM
from .train_text import (
    ByteTokenizer,
    _load_token_stream,
    _resolve_device,
    _sample_batch,
    _validate_training_config,
)


@dataclass(frozen=True)
class LossSummary:
    mean: float
    sample_std: float
    standard_error: float


@dataclass(frozen=True)
class PairedLossSummary:
    mean: float
    sample_std: float
    standard_error: float
    descriptive_normal_approx_95pct_interval: tuple[float, float]


@dataclass(frozen=True)
class BundleLossEvaluation:
    label: str
    bundle_path: str
    manifest_sha256: str
    config_sha256: str
    parameter_count: int
    total_loss: LossSummary
    causal_lm_loss: LossSummary
    mtp_loss: LossSummary | None


@dataclass(frozen=True)
class BundleComparisonReport:
    schema_version: int
    evaluator_sha256: str
    source: str
    corpus_sha256: str
    python_version: str
    torch_version: str
    machine: str
    device: str
    validation_seed: int
    batches: int
    batch_size: int
    context_length: int
    left: BundleLossEvaluation
    right: BundleLossEvaluation
    paired_left_minus_right_total: PairedLossSummary
    paired_left_minus_right_causal_lm: PairedLossSummary
    paired_left_minus_right_mtp: PairedLossSummary | None
    interval_kind: str
    independent_window_assumption: bool
    training_seed_variance_measured: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RawLosses:
    total: list[float]
    causal_lm: list[float]
    mtp: list[float] | None


def _evaluator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _summarize(values: list[float]) -> LossSummary:
    if len(values) < 2:
        raise ValueError("at least two observations are required for a loss summary.")
    sample_std = statistics.stdev(values)
    return LossSummary(
        mean=statistics.fmean(values),
        sample_std=sample_std,
        standard_error=sample_std / math.sqrt(len(values)),
    )


def _summarize_paired(left: list[float], right: list[float]) -> PairedLossSummary:
    if len(left) != len(right):
        raise ValueError("paired loss arrays must have the same length.")
    differences = [
        left_value - right_value
        for left_value, right_value in zip(left, right, strict=True)
    ]
    summary = _summarize(differences)
    radius = 1.96 * summary.standard_error
    return PairedLossSummary(
        mean=summary.mean,
        sample_std=summary.sample_std,
        standard_error=summary.standard_error,
        descriptive_normal_approx_95pct_interval=(
            summary.mean - radius,
            summary.mean + radius,
        ),
    )


def _validate_positive_integer(name: str, value: int, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")


def _verify_bundle(path: str | Path, label: str) -> PretrainedBundleReport:
    report = verify_deepseek_v4_pretrained_bundle(path)
    if not report.is_complete:
        details = "; ".join(report.errors) or "unknown verification error"
        raise ValueError(f"{label} bundle failed verification: {details}")
    return report


def _load_comparison_config(
    path: str | Path,
    *,
    label: str,
    tokenizer: ByteTokenizer,
    context_length: int,
) -> DeepSeekV4Config:
    config = DeepSeekV4Config.from_json_file(Path(path) / "config.json")
    try:
        _validate_training_config(config, tokenizer, context_length)
    except ValueError as exc:
        raise ValueError(f"{label} bundle is incompatible with byte-text comparison: {exc}") from exc
    return config


@torch.no_grad()
def _evaluate_bundle(
    *,
    path: str | Path,
    label: str,
    verification: PretrainedBundleReport,
    validation_tokens: torch.Tensor,
    context_length: int,
    batch_size: int,
    batches: int,
    validation_seed: int,
    device: torch.device,
) -> tuple[BundleLossEvaluation, _RawLosses]:
    # _verify_bundle already read and authenticated every bundle byte. Avoid a
    # second full checksum pass while the progressive loader materializes it.
    model = DeepSeekV4ForCausalLM.from_pretrained(
        path,
        device=device,
        verify_checksums=False,
    )
    model.eval()
    generator = torch.Generator().manual_seed(validation_seed)
    total_losses: list[float] = []
    causal_losses: list[float] = []
    mtp_losses: list[float] | None = [] if model.config.num_nextn_predict_layers else None
    try:
        for _ in range(batches):
            batch = _sample_batch(
                validation_tokens,
                context_length=context_length,
                batch_size=batch_size,
                generator=generator,
                device=device,
            )
            output = model(batch, labels=batch)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(f"{label} bundle produced a non-finite total loss.")
            total = float(output.loss)
            if mtp_losses is None:
                if output.mtp_loss is not None:
                    raise RuntimeError(f"{label} bundle returned unexpected MTP loss.")
                causal = total
            else:
                if output.mtp_loss is None or not torch.isfinite(output.mtp_loss):
                    raise RuntimeError(f"{label} bundle produced a missing or non-finite MTP loss.")
                mtp = float(output.mtp_loss)
                mtp_losses.append(mtp)
                causal = total - model.config.mtp_loss_weight * mtp
            total_losses.append(total)
            causal_losses.append(causal)

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        evaluation = BundleLossEvaluation(
            label=label,
            bundle_path=str(path),
            manifest_sha256=verification.manifest_sha256,
            config_sha256=verification.config_sha256,
            parameter_count=parameter_count,
            total_loss=_summarize(total_losses),
            causal_lm_loss=_summarize(causal_losses),
            mtp_loss=None if mtp_losses is None else _summarize(mtp_losses),
        )
        return evaluation, _RawLosses(
            total=total_losses,
            causal_lm=causal_losses,
            mtp=mtp_losses,
        )
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def compare_pretrained_bundles(
    *,
    text_file: str | Path,
    left_bundle: str | Path,
    right_bundle: str | Path,
    left_label: str = "left",
    right_label: str = "right",
    batches: int = 256,
    batch_size: int = 4,
    context_length: int = 128,
    validation_seed: int = 1338,
    device: str | torch.device = "auto",
) -> BundleComparisonReport:
    """Compare two byte-text bundles on identical held-out windows.

    The reported normal-approximation interval is descriptive only. Sampled
    byte windows may overlap, and this function does not measure training-seed
    variance.
    """

    _validate_positive_integer("batches", batches, minimum=2)
    _validate_positive_integer("batch_size", batch_size)
    _validate_positive_integer("context_length", context_length, minimum=4)
    if (
        isinstance(validation_seed, bool)
        or not isinstance(validation_seed, int)
        or validation_seed < 0
    ):
        raise ValueError("validation_seed must be a non-negative integer.")
    if not isinstance(left_label, str) or not left_label.strip():
        raise ValueError("left_label must be non-empty text.")
    if not isinstance(right_label, str) or not right_label.strip():
        raise ValueError("right_label must be non-empty text.")
    if left_label == right_label:
        raise ValueError("left_label and right_label must differ.")

    target_device = _resolve_device(device)
    tokenizer = ByteTokenizer()
    left_verification = _verify_bundle(left_bundle, left_label)
    right_verification = _verify_bundle(right_bundle, right_label)
    left_config = _load_comparison_config(
        left_bundle,
        label=left_label,
        tokenizer=tokenizer,
        context_length=context_length,
    )
    right_config = _load_comparison_config(
        right_bundle,
        label=right_label,
        tokenizer=tokenizer,
        context_length=context_length,
    )
    if left_config.num_nextn_predict_layers != right_config.num_nextn_predict_layers:
        raise ValueError("bundles must have the same number of MTP modules.")
    if left_config.mtp_loss_weight != right_config.mtp_loss_weight:
        raise ValueError("bundles must use the same mtp_loss_weight.")

    _, validation_tokens, source, corpus_sha256 = _load_token_stream(
        tokenizer,
        text_file,
        context_length,
    )
    left, left_raw = _evaluate_bundle(
        path=left_bundle,
        label=left_label,
        verification=left_verification,
        validation_tokens=validation_tokens,
        context_length=context_length,
        batch_size=batch_size,
        batches=batches,
        validation_seed=validation_seed,
        device=target_device,
    )
    right, right_raw = _evaluate_bundle(
        path=right_bundle,
        label=right_label,
        verification=right_verification,
        validation_tokens=validation_tokens,
        context_length=context_length,
        batch_size=batch_size,
        batches=batches,
        validation_seed=validation_seed,
        device=target_device,
    )
    if (left_raw.mtp is None) != (right_raw.mtp is None):
        raise RuntimeError("bundle MTP loss presence changed after configuration validation.")
    paired_mtp = (
        None
        if left_raw.mtp is None or right_raw.mtp is None
        else _summarize_paired(left_raw.mtp, right_raw.mtp)
    )
    return BundleComparisonReport(
        schema_version=1,
        evaluator_sha256=_evaluator_sha256(),
        source=source,
        corpus_sha256=corpus_sha256,
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        machine=platform.machine(),
        device=str(target_device),
        validation_seed=validation_seed,
        batches=batches,
        batch_size=batch_size,
        context_length=context_length,
        left=left,
        right=right,
        paired_left_minus_right_total=_summarize_paired(left_raw.total, right_raw.total),
        paired_left_minus_right_causal_lm=_summarize_paired(
            left_raw.causal_lm,
            right_raw.causal_lm,
        ),
        paired_left_minus_right_mtp=paired_mtp,
        interval_kind="descriptive_normal_approximation_over_overlapping_windows",
        independent_window_assumption=False,
        training_seed_variance_measured=False,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two native bundles on identical held-out byte windows.",
    )
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--left-bundle", type=Path, required=True)
    parser.add_argument("--right-bundle", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--batches", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--validation-seed", type=int, default=1338)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = compare_pretrained_bundles(
            text_file=args.text_file,
            left_bundle=args.left_bundle,
            right_bundle=args.right_bundle,
            left_label=args.left_label,
            right_label=args.right_label,
            batches=args.batches,
            batch_size=args.batch_size,
            context_length=args.context_length,
            validation_seed=args.validation_seed,
            device=args.device,
        )
        payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(payload, end="")
    else:
        print("nano-deepseek-v4 bundle comparison")
        print(f"corpus: {report.source} ({report.corpus_sha256})")
        print(
            f"{report.left.label}: total={report.left.total_loss.mean:.6f}, "
            f"causal={report.left.causal_lm_loss.mean:.6f}"
        )
        print(
            f"{report.right.label}: total={report.right.total_loss.mean:.6f}, "
            f"causal={report.right.causal_lm_loss.mean:.6f}"
        )
        paired = report.paired_left_minus_right_total
        lower, upper = paired.descriptive_normal_approx_95pct_interval
        print(
            f"paired {report.left.label} - {report.right.label}: "
            f"{paired.mean:+.6f} [{lower:+.6f}, {upper:+.6f}]"
        )
        print("interval: descriptive only; windows may overlap; training seeds not measured")
        print(f"evaluator sha256: {report.evaluator_sha256}")
        if args.output is not None:
            print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
