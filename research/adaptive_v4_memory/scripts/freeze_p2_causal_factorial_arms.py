from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from nano_deepseek_v4 import SameTokenControllerConfig, TrainingFreeControllerConfig

QuotaSource = Literal["uniform", "calibrated", "shuffled"]


@dataclass(frozen=True)
class CausalArmSpec:
    name: str
    quota_source: QuotaSource
    score_concentration: bool
    temporal_reuse: bool
    cross_layer_signal: bool
    refresh_reuse: bool
    protected_pins: bool
    dense_fallback: bool = False


@dataclass(frozen=True)
class BuiltCausalArm:
    spec: CausalArmSpec
    configs: tuple[SameTokenControllerConfig, ...]
    mixture_high_numerator: int = 0
    mixture_denominator: int = 1

    def __post_init__(self) -> None:
        if len(self.configs) not in {1, 2}:
            raise ValueError("A causal arm requires one config or a low/high pair.")
        if not 0 <= self.mixture_high_numerator < self.mixture_denominator:
            raise ValueError("The fixed-mixture fraction is invalid.")
        if len(self.configs) == 1 and self.mixture_high_numerator != 0:
            raise ValueError("A single-config arm cannot have a mixture fraction.")
        if len(self.configs) == 2 and self.mixture_high_numerator == 0:
            raise ValueError("A low/high arm requires a non-zero high fraction.")

    def config_for_batch(self, batch_index: int) -> SameTokenControllerConfig:
        """Choose low/high fixed top-k by a deterministic Bresenham schedule."""

        if batch_index < 0:
            raise ValueError("batch_index must be non-negative.")
        if len(self.configs) == 1:
            return self.configs[0]
        before = batch_index * self.mixture_high_numerator // self.mixture_denominator
        after = (batch_index + 1) * self.mixture_high_numerator // self.mixture_denominator
        return self.configs[1] if after > before else self.configs[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": asdict(self.spec),
            "configs": [asdict(config) for config in self.configs],
            "mixture_high_numerator": self.mixture_high_numerator,
            "mixture_denominator": self.mixture_denominator,
            "mixture_unit": "paired evaluation batch",
        }


PRIMARY_ARMS = (
    CausalArmSpec("fixed", "uniform", False, False, False, False, False),
    CausalArmSpec("fixed+pins", "uniform", False, False, False, False, True),
    CausalArmSpec("calibrated-no-pins", "calibrated", False, False, False, False, False),
    CausalArmSpec("calibrated+pins", "calibrated", False, False, False, False, True),
    CausalArmSpec("shuffled-quota", "shuffled", False, False, False, False, False),
    CausalArmSpec("shuffled-quota+pins", "shuffled", False, False, False, False, True),
    CausalArmSpec("local-no-pins", "calibrated", True, True, False, True, False),
    CausalArmSpec("local+pins", "calibrated", True, True, False, True, True),
    CausalArmSpec("hierarchical-no-pins", "calibrated", True, True, True, True, False),
    CausalArmSpec("hierarchical+pins", "calibrated", True, True, True, True, True),
)

COMPONENT_ARMS = (
    CausalArmSpec("hierarchical+pins-no-score", "calibrated", False, True, True, True, True),
    CausalArmSpec("hierarchical+pins-no-temporal", "calibrated", True, False, True, True, True),
    CausalArmSpec("hierarchical+pins-no-refresh", "calibrated", True, True, True, False, True),
    CausalArmSpec("hierarchical+pins+fallback", "calibrated", True, True, True, True, True, True),
)


def _pairs(raw: object, *, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list.")
    result: list[tuple[int, int]] = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
        ):
            raise ValueError(f"{name} contains an invalid layer-budget pair.")
        result.append((item[0], item[1]))
    if len({layer for layer, _ in result}) != len(result):
        raise ValueError(f"{name} contains duplicate layers.")
    return tuple(sorted(result))


def shuffled_layer_budgets(
    budgets: tuple[tuple[int, int], ...], calibration_digest: str
) -> tuple[tuple[tuple[int, int], ...], int, bool]:
    """Apply the preregistered digest-derived cyclic layer shuffle."""

    if len(budgets) < 2:
        raise ValueError("Quota shuffling requires at least two eligible layers.")
    if len(calibration_digest) != 64:
        raise ValueError("Calibration digest must be a SHA-256 hexadecimal digest.")
    try:
        int(calibration_digest, 16)
    except ValueError as error:
        raise ValueError("Calibration digest must be hexadecimal.") from error
    layers = tuple(layer for layer, _ in budgets)
    values = tuple(budget for _, budget in budgets)
    start = 1 + int(calibration_digest[:8], 16) % (len(values) - 1)
    offsets = tuple(range(start, len(values))) + tuple(range(1, start))
    for offset in offsets:
        rotated = values[offset:] + values[:offset]
        candidate = tuple(zip(layers, rotated, strict=True))
        if candidate != budgets:
            return candidate, offset, False
    return budgets, 0, True


def build_arm_configs(
    calibration: dict[str, Any],
    budget_label: str,
    *,
    fixed_match: dict[str, Any] | None = None,
) -> tuple[dict[str, BuiltCausalArm], dict[str, Any]]:
    if budget_label not in {"2x", "4x"}:
        raise ValueError("The primary causal factorial is frozen to 2x and 4x.")
    if calibration.get("experiment_id") != "p1-layer-quota-calibration-pilot-v1":
        raise ValueError("A frozen P1 quota-calibration artifact is required.")
    item = calibration.get("calibrations", {}).get(budget_label)
    if not isinstance(item, dict):
        raise ValueError(f"Calibration does not contain {budget_label}.")
    signal = TrainingFreeControllerConfig(**item["signal_config"])
    quota = item["quota"]
    calibrated = _pairs(quota["layer_budgets"], name="calibrated layer budgets")
    calibrated_total = sum(value for _, value in calibrated)
    if calibrated_total > signal.global_block_budget:
        raise ValueError("Calibrated layer budgets exceed the global budget.")
    logical_uniform_low, logical_remainder = divmod(calibrated_total, len(calibrated))
    if fixed_match is None:
        uniform_low_value = logical_uniform_low
        uniform_high_value = logical_uniform_low + int(logical_remainder > 0)
        fixed_high_numerator = logical_remainder
        fixed_mixture_denominator = len(calibrated)
        fixed_match_source = "configured-total-default"
    else:
        match = fixed_match.get("matches", {}).get(budget_label)
        if not isinstance(match, dict):
            raise ValueError(f"Physical hot-memory match is missing {budget_label}.")
        if match.get("calibration_digest") != quota.get("calibration_digest"):
            raise ValueError("Physical hot-memory match calibration digest drifted.")
        if match.get("passed") is not True:
            raise ValueError("Physical hot-memory match did not pass its calibration gate.")
        raw_uniform_low = match.get("uniform_low_blocks_per_layer")
        raw_uniform_high = match.get("uniform_high_blocks_per_layer")
        raw_high_numerator = match.get("mixture_high_numerator")
        raw_mixture_denominator = match.get("mixture_denominator")
        values = (
            raw_uniform_low,
            raw_uniform_high,
            raw_high_numerator,
            raw_mixture_denominator,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("Physical hot-memory match contains non-integer schedule values.")
        uniform_low_value = cast(int, raw_uniform_low)
        uniform_high_value = cast(int, raw_uniform_high)
        fixed_high_numerator = cast(int, raw_high_numerator)
        fixed_mixture_denominator = cast(int, raw_mixture_denominator)
        if (
            uniform_low_value <= 0
            or uniform_high_value < uniform_low_value
            or fixed_mixture_denominator <= 0
            or not 0 <= fixed_high_numerator < fixed_mixture_denominator
            or (uniform_high_value == uniform_low_value and fixed_high_numerator != 0)
            or (uniform_high_value > uniform_low_value and fixed_high_numerator == 0)
        ):
            raise ValueError("Physical hot-memory match contains an invalid fixed schedule.")
        fixed_match_source = "calibration-physical-hot-bytes"
    if uniform_low_value <= 0:
        raise ValueError("The calibrated total cannot preserve one fixed block per layer.")
    uniform_low = tuple((layer, uniform_low_value) for layer, _ in calibrated)
    uniform_high = tuple((layer, uniform_high_value) for layer, _ in calibrated)
    uniform_high_total = sum(value for _, value in uniform_high)
    fixed_signal = replace(
        signal,
        global_block_budget=max(signal.global_block_budget, uniform_high_total),
        dense_fallback_block_budget=max(signal.dense_fallback_block_budget, uniform_high_total),
    )
    digest = str(quota["calibration_digest"])
    shuffled, shuffle_offset, structurally_identical = shuffled_layer_budgets(calibrated, digest)
    quota_sources = {
        "calibrated": calibrated,
        "shuffled": shuffled,
    }

    def make_config(
        arm: CausalArmSpec,
        layer_budgets: tuple[tuple[int, int], ...],
        *,
        signal_config: TrainingFreeControllerConfig = signal,
    ) -> SameTokenControllerConfig:
        return SameTokenControllerConfig(
            signal=signal_config,
            layer_budgets=layer_budgets,
            dense_layer_budgets=layer_budgets,
            enable_score_concentration=arm.score_concentration,
            enable_temporal_reuse=arm.temporal_reuse,
            enable_cross_layer_signal=arm.cross_layer_signal,
            enable_refresh_reuse=arm.refresh_reuse,
            enable_protected_pins=arm.protected_pins,
            enable_dense_fallback=arm.dense_fallback,
        )

    configs: dict[str, BuiltCausalArm] = {}
    for arm in (*PRIMARY_ARMS, *COMPONENT_ARMS):
        if arm.quota_source == "uniform":
            low = make_config(arm, uniform_low, signal_config=fixed_signal)
            if fixed_high_numerator:
                configs[arm.name] = BuiltCausalArm(
                    spec=arm,
                    configs=(
                        low,
                        make_config(arm, uniform_high, signal_config=fixed_signal),
                    ),
                    mixture_high_numerator=fixed_high_numerator,
                    mixture_denominator=fixed_mixture_denominator,
                )
            else:
                configs[arm.name] = BuiltCausalArm(spec=arm, configs=(low,))
        else:
            configs[arm.name] = BuiltCausalArm(
                spec=arm,
                configs=(make_config(arm, quota_sources[arm.quota_source]),),
            )
    metadata = {
        "budget_label": budget_label,
        "calibration_digest": digest,
        "calibrated_total_blocks": calibrated_total,
        "fixed_uniform_low_layer_budgets": uniform_low,
        "fixed_uniform_high_layer_budgets": (uniform_high if fixed_high_numerator else uniform_low),
        "fixed_mixture_high_numerator": fixed_high_numerator,
        "fixed_mixture_denominator": fixed_mixture_denominator,
        "fixed_match_source": fixed_match_source,
        "fixed_controller_validation_ceiling": fixed_signal.global_block_budget,
        "fixed_controller_validation_ceiling_adjusted": (
            fixed_signal.global_block_budget != signal.global_block_budget
        ),
        "calibrated_layer_budgets": calibrated,
        "shuffled_layer_budgets": shuffled,
        "shuffle_offset": shuffle_offset,
        "shuffled_is_structurally_identical": structurally_identical,
        "all_nonfixed_primary_arms_share_calibrated_configured_total": all(
            sum(value for _, value in configs[arm.name].configs[0].layer_budgets)
            == calibrated_total
            for arm in PRIMARY_ARMS
            if arm.quota_source != "uniform"
        ),
        "fixed_mixture_mean_configured_total": (
            (
                (fixed_mixture_denominator - fixed_high_numerator)
                * sum(value for _, value in uniform_low)
                + fixed_high_numerator * sum(value for _, value in uniform_high)
            )
            / fixed_mixture_denominator
        ),
    }
    return configs, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one P2 causal-factorial arm set.")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--budget", choices=("2x", "4x"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("research/adaptive_v4_memory/manifests/p2-causal-factorial-v1.json"),
    )
    args = parser.parse_args()
    source = _source_state()
    if source["dirty"]:
        raise RuntimeError("Causal arm freezing requires a clean source tree.")
    design = json.loads(args.manifest.read_text())
    if design.get("experiment_id") != "p2-causal-factorial-v1":
        raise ValueError("The frozen P2 causal-factorial manifest is required.")
    if set(design.get("primary_arms", {})) != {arm.name for arm in PRIMARY_ARMS}:
        raise ValueError("Causal-factorial manifest arm set drifted.")
    if args.budget not in design.get("central_gate", {}).get("required_budget_points", ()):
        raise ValueError("Requested budget is outside the central causal gate.")
    calibration = json.loads(args.calibration.read_text())
    configs, metadata = build_arm_configs(calibration, args.budget)
    payload = {
        "schema_version": 1,
        "experiment_id": "p2-causal-factorial-arm-freeze-v1",
        "source": source,
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "design_manifest": {
            "path": str(args.manifest),
            "sha256": _sha256(args.manifest),
        },
        "calibration": {
            "path": str(args.calibration),
            "sha256": _sha256(args.calibration),
            "scale": calibration.get("scale"),
            "training_seed": calibration.get("training_seed"),
            "calibration_seed": calibration.get("seed"),
        },
        "metadata": metadata,
        "primary_arm_order": [arm.name for arm in PRIMARY_ARMS],
        "component_arm_order": [arm.name for arm in COMPONENT_ARMS],
        "configs": {name: arm.to_dict() for name, arm in configs.items()},
        "claim_boundary": "Configuration freeze only; no held-out quality result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "metadata": metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
