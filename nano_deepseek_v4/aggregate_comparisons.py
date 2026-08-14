from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .compare_bundles import _evaluator_sha256


@dataclass(frozen=True)
class RunEffect:
    run_id: str
    report_path: str
    report_sha256: str
    left_manifest_sha256: str
    right_manifest_sha256: str
    paired_left_minus_right_total: float
    paired_left_minus_right_causal_lm: float
    paired_left_minus_right_mtp: float | None


@dataclass(frozen=True)
class RunEffectSummary:
    observation_count: int
    mean: float
    sample_std: float
    standard_error: float
    minimum: float
    maximum: float
    negative_count: int
    zero_count: int
    positive_count: int


@dataclass(frozen=True)
class BundleComparisonAggregate:
    schema_version: int
    aggregator_sha256: str
    comparison_evaluator_sha256: str
    comparison_evaluator_matches_installed: bool
    corpus_sha256: str
    python_version: str
    torch_version: str
    machine: str
    device: str
    validation_seed: int
    batches: int
    batch_size: int
    context_length: int
    left_label: str
    right_label: str
    left_config_sha256: str
    right_config_sha256: str
    left_parameter_count: int
    right_parameter_count: int
    run_ids: tuple[str, ...]
    run_count: int
    runs: tuple[RunEffect, ...]
    paired_left_minus_right_total: RunEffectSummary
    paired_left_minus_right_causal_lm: RunEffectSummary
    paired_left_minus_right_mtp: RunEffectSummary | None
    evaluation_window_independence_assumed: bool
    run_independence_verified: bool
    training_seed_variance_measured: bool
    claim_boundary: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ComparisonContract:
    evaluator_sha256: str
    corpus_sha256: str
    python_version: str
    torch_version: str
    machine: str
    device: str
    validation_seed: int
    batches: int
    batch_size: int
    context_length: int
    left_label: str
    right_label: str
    left_config_sha256: str
    right_config_sha256: str
    left_parameter_count: int
    right_parameter_count: int
    interval_kind: str
    independent_window_assumption: bool
    has_mtp: bool


def _aggregator_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object.")
    return value


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text.")
    return value


def _require_sha256(value: Any, field: str) -> str:
    digest = _require_text(value, field).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest.")
    return digest


def _require_integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer of at least {minimum}.")
    return value


def _require_boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    return result


def _paired_mean(report: Mapping[str, Any], field: str) -> float | None:
    value = report.get(field)
    if value is None:
        return None
    summary = _require_mapping(value, field)
    return _require_finite_number(summary.get("mean"), f"{field}.mean")


def _load_run(
    run_id: str,
    path: Path,
) -> tuple[RunEffect, _ComparisonContract]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read comparison report {path}: {exc}") from exc
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"comparison report {path} is not valid UTF-8 JSON: {exc}") from exc
    report = _require_mapping(parsed, str(path))
    if _require_integer(report.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError(f"comparison report {path} has an unsupported schema_version.")
    if _require_boolean(
        report.get("training_seed_variance_measured"),
        "training_seed_variance_measured",
    ):
        raise ValueError(f"comparison report {path} is not a single-run report.")

    left = _require_mapping(report.get("left"), "left")
    right = _require_mapping(report.get("right"), "right")
    total = _paired_mean(report, "paired_left_minus_right_total")
    causal = _paired_mean(report, "paired_left_minus_right_causal_lm")
    mtp = _paired_mean(report, "paired_left_minus_right_mtp")
    if total is None or causal is None:
        raise ValueError(f"comparison report {path} is missing required paired losses.")
    left_mtp = left.get("mtp_loss")
    right_mtp = right.get("mtp_loss")
    if (left_mtp is None) != (right_mtp is None) or (mtp is None) != (left_mtp is None):
        raise ValueError(f"comparison report {path} has inconsistent MTP summaries.")

    contract = _ComparisonContract(
        evaluator_sha256=_require_sha256(report.get("evaluator_sha256"), "evaluator_sha256"),
        corpus_sha256=_require_sha256(report.get("corpus_sha256"), "corpus_sha256"),
        python_version=_require_text(report.get("python_version"), "python_version"),
        torch_version=_require_text(report.get("torch_version"), "torch_version"),
        machine=_require_text(report.get("machine"), "machine"),
        device=_require_text(report.get("device"), "device"),
        validation_seed=_require_integer(report.get("validation_seed"), "validation_seed"),
        batches=_require_integer(report.get("batches"), "batches", minimum=2),
        batch_size=_require_integer(report.get("batch_size"), "batch_size", minimum=1),
        context_length=_require_integer(
            report.get("context_length"),
            "context_length",
            minimum=4,
        ),
        left_label=_require_text(left.get("label"), "left.label"),
        right_label=_require_text(right.get("label"), "right.label"),
        left_config_sha256=_require_sha256(left.get("config_sha256"), "left.config_sha256"),
        right_config_sha256=_require_sha256(
            right.get("config_sha256"),
            "right.config_sha256",
        ),
        left_parameter_count=_require_integer(
            left.get("parameter_count"),
            "left.parameter_count",
            minimum=1,
        ),
        right_parameter_count=_require_integer(
            right.get("parameter_count"),
            "right.parameter_count",
            minimum=1,
        ),
        interval_kind=_require_text(report.get("interval_kind"), "interval_kind"),
        independent_window_assumption=_require_boolean(
            report.get("independent_window_assumption"),
            "independent_window_assumption",
        ),
        has_mtp=mtp is not None,
    )
    effect = RunEffect(
        run_id=run_id,
        report_path=str(path),
        report_sha256=hashlib.sha256(payload).hexdigest(),
        left_manifest_sha256=_require_sha256(
            left.get("manifest_sha256"),
            "left.manifest_sha256",
        ),
        right_manifest_sha256=_require_sha256(
            right.get("manifest_sha256"),
            "right.manifest_sha256",
        ),
        paired_left_minus_right_total=total,
        paired_left_minus_right_causal_lm=causal,
        paired_left_minus_right_mtp=mtp,
    )
    return effect, contract


def _sqrt_fraction_as_float(value: Fraction) -> float:
    """Return the correctly rounded binary64 square root of a nonnegative ratio."""

    if value < 0:
        raise ValueError("cannot take the square root of a negative fraction.")
    numerator = value.numerator
    denominator = value.denominator
    rounding_bits = 2 * sys.float_info.mant_dig + 3
    shift = (numerator.bit_length() - denominator.bit_length() - rounding_bits) // 2
    if shift >= 0:
        scaled_denominator = denominator << (2 * shift)
        root = math.isqrt(numerator // scaled_denominator)
        rounded_to_odd = root | (root * root * scaled_denominator != numerator)
        return float(rounded_to_odd << shift)

    scaled_numerator = numerator << (-2 * shift)
    root = math.isqrt(scaled_numerator // denominator)
    rounded_to_odd = root | (root * root * denominator != scaled_numerator)
    return rounded_to_odd / (1 << (-shift))


def _summarize(values: list[float]) -> RunEffectSummary:
    if len(values) < 2:
        raise ValueError("at least two observations are required.")
    exact_values = [Fraction.from_float(value) for value in values]
    count = len(exact_values)
    exact_mean = sum(exact_values, start=Fraction()) / count
    sample_variance = (
        sum(((value - exact_mean) ** 2 for value in exact_values), start=Fraction())
        / (count - 1)
    )
    sample_std = _sqrt_fraction_as_float(sample_variance)
    try:
        mean = statistics.fmean(values)
    except OverflowError:
        mean = float(exact_mean)
    return RunEffectSummary(
        observation_count=count,
        mean=mean,
        sample_std=sample_std,
        standard_error=sample_std / math.sqrt(count),
        minimum=min(values),
        maximum=max(values),
        negative_count=sum(value < 0 for value in values),
        zero_count=sum(value == 0 for value in values),
        positive_count=sum(value > 0 for value in values),
    )


def aggregate_bundle_comparison_reports(
    reports: Sequence[tuple[str, str | Path]],
) -> BundleComparisonAggregate:
    """Validate and summarize compatible single-run bundle comparisons.

    Run identifiers are user-supplied labels. This function verifies the
    evaluation contract, but cannot prove that bundles came from independent
    training runs or that a run identifier is a training seed.
    """

    if len(reports) < 2:
        raise ValueError("at least two comparison reports are required.")
    normalized: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for run_id, path in reports:
        run_id = _require_text(run_id, "run_id")
        if run_id in seen:
            raise ValueError(f"duplicate run_id: {run_id}")
        seen.add(run_id)
        normalized.append((run_id, Path(path)))

    effects: list[RunEffect] = []
    reference_contract: _ComparisonContract | None = None
    for run_id, path in sorted(normalized):
        effect, contract = _load_run(run_id, path)
        if reference_contract is None:
            reference_contract = contract
        elif contract != reference_contract:
            changed = [
                field
                for field in contract.__dataclass_fields__
                if getattr(contract, field) != getattr(reference_contract, field)
            ]
            raise ValueError(
                f"comparison report {path} changes the common contract: {', '.join(changed)}"
            )
        effects.append(effect)
    assert reference_contract is not None

    bundle_manifests = [
        manifest
        for effect in effects
        for manifest in (
            effect.left_manifest_sha256,
            effect.right_manifest_sha256,
        )
    ]
    if len(set(bundle_manifests)) != len(bundle_manifests):
        raise ValueError("each run must reference distinct left and right bundle manifests.")

    total_values = [effect.paired_left_minus_right_total for effect in effects]
    causal_values = [effect.paired_left_minus_right_causal_lm for effect in effects]
    mtp_values = [effect.paired_left_minus_right_mtp for effect in effects]
    mtp_summary = None
    if reference_contract.has_mtp:
        if any(value is None for value in mtp_values):
            raise RuntimeError("MTP presence changed after contract validation.")
        mtp_summary = _summarize([float(value) for value in mtp_values if value is not None])

    return BundleComparisonAggregate(
        schema_version=1,
        aggregator_sha256=_aggregator_sha256(),
        comparison_evaluator_sha256=reference_contract.evaluator_sha256,
        comparison_evaluator_matches_installed=(
            reference_contract.evaluator_sha256 == _evaluator_sha256()
        ),
        corpus_sha256=reference_contract.corpus_sha256,
        python_version=reference_contract.python_version,
        torch_version=reference_contract.torch_version,
        machine=reference_contract.machine,
        device=reference_contract.device,
        validation_seed=reference_contract.validation_seed,
        batches=reference_contract.batches,
        batch_size=reference_contract.batch_size,
        context_length=reference_contract.context_length,
        left_label=reference_contract.left_label,
        right_label=reference_contract.right_label,
        left_config_sha256=reference_contract.left_config_sha256,
        right_config_sha256=reference_contract.right_config_sha256,
        left_parameter_count=reference_contract.left_parameter_count,
        right_parameter_count=reference_contract.right_parameter_count,
        run_ids=tuple(effect.run_id for effect in effects),
        run_count=len(effects),
        runs=tuple(effects),
        paired_left_minus_right_total=_summarize(total_values),
        paired_left_minus_right_causal_lm=_summarize(causal_values),
        paired_left_minus_right_mtp=mtp_summary,
        evaluation_window_independence_assumed=(
            reference_contract.independent_window_assumption
        ),
        run_independence_verified=False,
        training_seed_variance_measured=False,
        claim_boundary=(
            "Run identifiers are user-supplied; this aggregate does not prove independent training.",
            "Run-level summaries are descriptive and do not establish architecture superiority.",
            "Evaluation-window dependence remains as declared by the source comparisons.",
        ),
    )


def _parse_report_spec(value: str) -> tuple[str, Path]:
    run_id, separator, path = value.partition("=")
    if not separator or not run_id.strip() or not path.strip():
        raise argparse.ArgumentTypeError("report must use RUN_ID=PATH syntax.")
    return run_id, Path(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate compatible bundle-comparison reports across runs.",
    )
    parser.add_argument(
        "--report",
        action="append",
        type=_parse_report_spec,
        required=True,
        metavar="RUN_ID=PATH",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = aggregate_bundle_comparison_reports(args.report)
        payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(payload, end="")
    else:
        print("nano-deepseek-v4 comparison aggregate")
        print(f"runs: {', '.join(report.run_ids)}")
        paired = report.paired_left_minus_right_total
        print(
            f"paired {report.left_label} - {report.right_label}: "
            f"mean={paired.mean:+.6f}, sd={paired.sample_std:.6f}, "
            f"range=[{paired.minimum:+.6f}, {paired.maximum:+.6f}]"
        )
        print("run independence: not verified; run IDs are user-supplied")
        print(f"aggregator sha256: {report.aggregator_sha256}")
        if args.output is not None:
            print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
