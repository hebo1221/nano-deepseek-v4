from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM

_SCHEMA_VERSION = 2
_PROTOCOL_NAME = "native-attention-path-reachability"
_SEQUENCE_LENGTH = 256
_QUERY_POSITION = 255
_DISTANCES = (16, 64, 94, 95, 128, 224)
_LOCAL_WINDOW = 32
_FULL_WINDOW = 256
_NUM_LAYERS = 3
_DEFAULT_MODEL_SEED = 1337
_DEFAULT_INPUT_SEED = 20_260_811
_DEFAULT_GRADIENT_SEED = 424_242
_DEFAULT_PROBE_COUNT = 5
_MAX_PROBE_COUNT = 64
_FIRST_PROBE_TOKEN = 3
_VOCAB_SIZE = 64
_INFLUENCE_ATOL = 1e-6


@dataclass(frozen=True)
class ReachObservation:
    distance: int
    source_position: int
    source_lag: int
    theoretically_reachable: bool
    expected_changed: bool
    exactly_changed_probe_count: int
    materially_changed_probe_count: int
    probe_count: int
    min_linf_delta: float
    max_linf_delta: float
    mean_linf_delta: float
    max_l2_delta: float
    mean_l2_delta: float
    passed: bool


@dataclass(frozen=True)
class ReachGradientObservation:
    distance: int
    source_position: int
    source_lag: int
    theoretically_reachable: bool
    expected_nonzero: bool
    nonzero_probe_count: int
    probe_count: int
    min_l2_gradient: float
    max_l2_gradient: float
    mean_l2_gradient: float
    passed: bool


@dataclass(frozen=True)
class ReachVariantResult:
    name: str
    attention_schedule: tuple[str, ...]
    sliding_window: int
    parameter_count: int
    config_sha256: str
    perturbation_observations: tuple[ReachObservation, ...]
    gradient_observations: tuple[ReachGradientObservation, ...]
    perturbation_passed: bool
    gradient_passed: bool
    passed: bool


@dataclass(frozen=True)
class AttentionReachReport:
    schema_version: int
    name: str
    protocol_sha256: str
    source_sha256: dict[str, str]
    python_version: str
    torch_version: str
    device: str
    device_name: str
    model_seed: int
    input_seed: int
    gradient_seed: int
    probe_count: int
    probe_input_sha256: str
    mutation_batch_sha256: str
    gradient_cotangent_sha256: str
    sequence_length: int
    query_position: int
    distance_definition: str
    distances: tuple[int, ...]
    local_window: int
    local_max_source_lag: int
    first_disconnected_distance: int
    influence_atol: float
    gradient_objective: str
    local_full_config_difference: tuple[str, ...]
    local_full_state_dict_identical: bool
    hybrid_minus_local_parameter_count: int
    hybrid_parameter_delta_fraction: float
    variants: dict[str, ReachVariantResult]
    claim_boundary: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
    }
    digest = hashlib.sha256(_canonical_json_bytes(descriptor))
    digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _protocol(configs: Mapping[str, DeepSeekV4Config]) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "name": _PROTOCOL_NAME,
        "sequence_length": _SEQUENCE_LENGTH,
        "query_position": _QUERY_POSITION,
        "distance_definition": "one_based: source_lag = distance - 1",
        "distances": list(_DISTANCES),
        "local_receptive_field": {
            "layers": _NUM_LAYERS,
            "window": _LOCAL_WINDOW,
            "max_source_lag": _NUM_LAYERS * (_LOCAL_WINDOW - 1),
        },
        "mutation": {
            "token_interval": [_FIRST_PROBE_TOKEN, _VOCAB_SIZE - 1],
            "rule": "next token modulo the non-special probe-token interval",
        },
        "defaults": {
            "model_seed": _DEFAULT_MODEL_SEED,
            "input_seed": _DEFAULT_INPUT_SEED,
            "gradient_seed": _DEFAULT_GRADIENT_SEED,
            "probe_count": _DEFAULT_PROBE_COUNT,
        },
        "variants": {
            name: {
                "config_sha256": _config_sha256(config),
                "attention_schedule": list(config.layer_types or ()),
                "mlp_schedule": list(config.mlp_layer_types or ()),
                "sliding_window": config.sliding_window,
            }
            for name, config in configs.items()
        },
        "pass_contract": {
            "token_perturbation": {
                "influence_atol": _INFLUENCE_ATOL,
                "hybrid": (
                    "every probe exceeds the influence tolerance at every pinned distance"
                ),
                "local": (
                    "every probe exceeds the influence tolerance through lag 93 and no "
                    "probe exceeds it after lag 93"
                ),
                "full_window": (
                    "every probe exceeds the influence tolerance at every pinned distance"
                ),
            },
            "embedding_gradient": {
                "objective": "fixed Rademacher VJP of final-position logits",
                "hybrid": "every probe has a nonzero gradient at every pinned distance",
                "local": (
                    "every probe has a nonzero gradient through lag 93 and an exact-zero "
                    "gradient after lag 93"
                ),
                "full_window": (
                    "every probe has a nonzero gradient at every pinned distance"
                ),
            },
        },
    }


def _common_config_kwargs() -> dict[str, Any]:
    return {
        "vocab_size": _VOCAB_SIZE,
        "hidden_size": 96,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 24,
        "q_lora_rank": 48,
        "num_experts_per_tok": 2,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "max_position_embeddings": 512,
        "compress_rates": {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
        "num_hash_layers": 1,
        "mlp_layer_types": ["hash_moe", "moe", "moe"],
        "hc_mult": 2,
        "hc_sinkhorn_iters": 8,
        "o_groups": 2,
        "o_lora_rank": 24,
        "index_n_heads": 4,
        "index_head_dim": 12,
        "index_topk": 8,
        "num_nextn_predict_layers": 0,
        "mtp_layer_types": [],
        "attention_dropout": 0.0,
        "partial_rotary_factor": 0.5,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }


def build_attention_reach_configs() -> dict[str, DeepSeekV4Config]:
    """Build the fixed hybrid and near-parameter-matched control configs."""

    hybrid = DeepSeekV4Config(
        **_common_config_kwargs(),
        moe_intermediate_size=192,
        sliding_window=_LOCAL_WINDOW,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
    )
    local = DeepSeekV4Config(
        **_common_config_kwargs(),
        moe_intermediate_size=197,
        sliding_window=_LOCAL_WINDOW,
        layer_types=["sliding_attention"] * _NUM_LAYERS,
    )
    full_window = DeepSeekV4Config(
        **_common_config_kwargs(),
        moe_intermediate_size=197,
        sliding_window=_FULL_WINDOW,
        layer_types=["sliding_attention"] * _NUM_LAYERS,
    )
    return {
        "hybrid": hybrid,
        "local": local,
        "full_window": full_window,
    }


def _validate_seed(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError(f"{name} must be an integer in [0, 2**63).")


def _validate_probe_count(probe_count: int) -> None:
    if (
        isinstance(probe_count, bool)
        or not isinstance(probe_count, int)
        or not 1 <= probe_count <= _MAX_PROBE_COUNT
    ):
        raise ValueError(f"probe_count must be an integer in [1, {_MAX_PROBE_COUNT}].")


def _resolve_device(device: str | torch.device) -> torch.device:
    try:
        resolved = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc
    if resolved.type == "cpu":
        if resolved.index is not None:
            raise ValueError("indexed CPU devices are unsupported; use 'cpu'.")
        return resolved
    if resolved.type != "cuda":
        raise ValueError("device must be 'cpu', 'cuda', or a valid CUDA device index.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    index = torch.cuda.current_device() if resolved.index is None else resolved.index
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError(f"CUDA device index {index} is unavailable.")
    return torch.device("cuda", index)


def _build_probe_batch(
    *,
    input_seed: int,
    probe_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(input_seed)
    probes = torch.randint(
        _FIRST_PROBE_TOKEN,
        _VOCAB_SIZE,
        (probe_count, _SEQUENCE_LENGTH),
        generator=generator,
        dtype=torch.long,
    )
    rows: list[torch.Tensor] = []
    token_span = _VOCAB_SIZE - _FIRST_PROBE_TOKEN
    for probe in probes:
        rows.append(probe)
        for distance in _DISTANCES:
            source_position = _QUERY_POSITION - (distance - 1)
            mutated = probe.clone()
            token = int(mutated[source_position])
            mutated[source_position] = (
                _FIRST_PROBE_TOKEN + (token - _FIRST_PROBE_TOKEN + 1) % token_span
            )
            rows.append(mutated)
    return probes, torch.stack(rows)


def _build_gradient_cotangent(*, gradient_seed: int, probe_count: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(gradient_seed)
    signs = torch.randint(
        0,
        2,
        (probe_count, _VOCAB_SIZE),
        generator=generator,
        dtype=torch.int8,
    )
    return signs.to(dtype=torch.float32).mul_(2).sub_(1)


def _config_sha256(config: DeepSeekV4Config) -> str:
    return _sha256_json(config.to_dict())


def _parameter_count(model: DeepSeekV4ForCausalLM) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _state_dicts_identical(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> bool:
    return left.keys() == right.keys() and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _observe_variant(
    *,
    name: str,
    model: DeepSeekV4ForCausalLM,
    config: DeepSeekV4Config,
    perturbation_batch: torch.Tensor,
    probes: torch.Tensor,
    gradient_cotangent: torch.Tensor,
    device: torch.device,
    local_max_source_lag: int,
) -> ReachVariantResult:
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = (
            model(perturbation_batch.to(device))
            .logits[:, _QUERY_POSITION]
            .float()
            .cpu()
        )
    if not torch.isfinite(logits).all():
        raise RuntimeError(f"{name} produced non-finite perturbation logits.")

    grouped = logits.view(-1, len(_DISTANCES) + 1, logits.shape[-1])
    base = grouped[:, 0]
    perturbation_observations: list[ReachObservation] = []
    for offset, distance in enumerate(_DISTANCES, start=1):
        source_lag = distance - 1
        source_position = _QUERY_POSITION - source_lag
        delta = (grouped[:, offset] - base).abs()
        linf = delta.amax(dim=-1)
        l2 = torch.linalg.vector_norm(delta, dim=-1)
        exactly_changed = delta.ne(0).any(dim=-1)
        materially_changed = linf > _INFLUENCE_ATOL
        materially_changed_count = int(materially_changed.sum())
        theoretically_reachable = (
            source_lag <= local_max_source_lag if name == "local" else True
        )
        expected_changed = theoretically_reachable
        passed = (
            materially_changed_count == materially_changed.numel()
            if expected_changed
            else materially_changed_count == 0
        )
        perturbation_observations.append(
            ReachObservation(
                distance=distance,
                source_position=source_position,
                source_lag=source_lag,
                theoretically_reachable=theoretically_reachable,
                expected_changed=expected_changed,
                exactly_changed_probe_count=int(exactly_changed.sum()),
                materially_changed_probe_count=materially_changed_count,
                probe_count=materially_changed.numel(),
                min_linf_delta=float(linf.min()),
                max_linf_delta=float(linf.max()),
                mean_linf_delta=float(linf.mean()),
                max_l2_delta=float(l2.max()),
                mean_l2_delta=float(l2.mean()),
                passed=passed,
            )
        )

    captured_embeddings: list[torch.Tensor] = []

    def capture_embedding(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        leaf = output.detach().requires_grad_(True)
        captured_embeddings.append(leaf)
        return leaf

    parameter_count = _parameter_count(model)
    model.requires_grad_(False)
    hook = model.model.embed_tokens.register_forward_hook(capture_embedding)
    try:
        with torch.enable_grad():
            gradient_logits = model(probes.to(device)).logits[:, _QUERY_POSITION].float()
            if len(captured_embeddings) != 1:
                raise RuntimeError(
                    f"{name} embedding hook fired {len(captured_embeddings)} times; expected once."
                )
            if not torch.isfinite(gradient_logits).all():
                raise RuntimeError(f"{name} produced non-finite gradient-probe logits.")
            objective = (
                gradient_logits * gradient_cotangent.to(device=device)
            ).sum()
            (embedding_gradient,) = torch.autograd.grad(
                objective,
                captured_embeddings[0],
                create_graph=False,
                retain_graph=False,
            )
    finally:
        hook.remove()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError(f"{name} gradient probe populated parameter gradients.")
    if not torch.isfinite(embedding_gradient).all():
        raise RuntimeError(f"{name} produced non-finite embedding gradients.")

    gradient_observations: list[ReachGradientObservation] = []
    for distance in _DISTANCES:
        source_lag = distance - 1
        source_position = _QUERY_POSITION - source_lag
        gradient_norm = torch.linalg.vector_norm(
            embedding_gradient[:, source_position].float(),
            dim=-1,
        ).cpu()
        nonzero = gradient_norm.ne(0)
        nonzero_count = int(nonzero.sum())
        theoretically_reachable = (
            source_lag <= local_max_source_lag if name == "local" else True
        )
        expected_nonzero = theoretically_reachable
        passed = (
            nonzero_count == nonzero.numel()
            if expected_nonzero
            else nonzero_count == 0
        )
        gradient_observations.append(
            ReachGradientObservation(
                distance=distance,
                source_position=source_position,
                source_lag=source_lag,
                theoretically_reachable=theoretically_reachable,
                expected_nonzero=expected_nonzero,
                nonzero_probe_count=nonzero_count,
                probe_count=nonzero.numel(),
                min_l2_gradient=float(gradient_norm.min()),
                max_l2_gradient=float(gradient_norm.max()),
                mean_l2_gradient=float(gradient_norm.mean()),
                passed=passed,
            )
        )

    perturbation_passed = all(item.passed for item in perturbation_observations)
    gradient_passed = all(item.passed for item in gradient_observations)
    return ReachVariantResult(
        name=name,
        attention_schedule=tuple(config.layer_types or ()),
        sliding_window=config.sliding_window,
        parameter_count=parameter_count,
        config_sha256=_config_sha256(config),
        perturbation_observations=tuple(perturbation_observations),
        gradient_observations=tuple(gradient_observations),
        perturbation_passed=perturbation_passed,
        gradient_passed=gradient_passed,
        passed=perturbation_passed and gradient_passed,
    )


def _source_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    return {
        "nano_deepseek_v4/attention_reach.py": _sha256_file(Path(__file__).resolve()),
        "nano_deepseek_v4/config.py": _sha256_file(package_root / "config.py"),
        "nano_deepseek_v4/modeling.py": _sha256_file(package_root / "modeling.py"),
    }


def run_attention_reach(
    *,
    model_seed: int = _DEFAULT_MODEL_SEED,
    input_seed: int = _DEFAULT_INPUT_SEED,
    gradient_seed: int = _DEFAULT_GRADIENT_SEED,
    probe_count: int = _DEFAULT_PROBE_COUNT,
    device: str | torch.device = "cpu",
) -> AttentionReachReport:
    """Run the fixed structural influence check without training any model."""

    _validate_seed("model_seed", model_seed)
    _validate_seed("input_seed", input_seed)
    _validate_seed("gradient_seed", gradient_seed)
    _validate_probe_count(probe_count)
    resolved_device = _resolve_device(device)
    probes, mutation_batch = _build_probe_batch(
        input_seed=input_seed,
        probe_count=probe_count,
    )
    gradient_cotangent = _build_gradient_cotangent(
        gradient_seed=gradient_seed,
        probe_count=probe_count,
    )
    configs = build_attention_reach_configs()
    local_max_source_lag = _NUM_LAYERS * (_LOCAL_WINDOW - 1)
    with torch.random.fork_rng(devices=[]):
        models: dict[str, DeepSeekV4ForCausalLM] = {}
        for name in ("hybrid", "local", "full_window"):
            torch.random.default_generator.manual_seed(model_seed)
            models[name] = DeepSeekV4ForCausalLM(configs[name])

        control_state_identical = _state_dicts_identical(
            models["local"].state_dict(),
            models["full_window"].state_dict(),
        )
        variants = {
            name: _observe_variant(
                name=name,
                model=models[name],
                config=configs[name],
                perturbation_batch=mutation_batch,
                probes=probes,
                gradient_cotangent=gradient_cotangent,
                device=resolved_device,
                local_max_source_lag=local_max_source_lag,
            )
            for name in ("hybrid", "local", "full_window")
        }

    local_parameters = variants["local"].parameter_count
    hybrid_delta = variants["hybrid"].parameter_count - local_parameters
    config_diff = tuple(
        sorted(
            key
            for key in configs["local"].to_dict()
            if configs["local"].to_dict()[key] != configs["full_window"].to_dict()[key]
        )
    )
    passed = (
        control_state_identical
        and config_diff == ("sliding_window",)
        and variants["local"].parameter_count
        == variants["full_window"].parameter_count
        and hybrid_delta == 60
        and all(variant.passed for variant in variants.values())
    )
    device_name = (
        torch.cuda.get_device_name(resolved_device)
        if resolved_device.type == "cuda"
        else "CPU"
    )
    return AttentionReachReport(
        schema_version=_SCHEMA_VERSION,
        name=_PROTOCOL_NAME,
        protocol_sha256=_sha256_json(_protocol(configs)),
        source_sha256=_source_hashes(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        device=str(resolved_device),
        device_name=device_name,
        model_seed=model_seed,
        input_seed=input_seed,
        gradient_seed=gradient_seed,
        probe_count=probe_count,
        probe_input_sha256=_tensor_sha256(probes),
        mutation_batch_sha256=_tensor_sha256(mutation_batch),
        gradient_cotangent_sha256=_tensor_sha256(gradient_cotangent),
        sequence_length=_SEQUENCE_LENGTH,
        query_position=_QUERY_POSITION,
        distance_definition="one_based: source_lag = distance - 1",
        distances=_DISTANCES,
        local_window=_LOCAL_WINDOW,
        local_max_source_lag=local_max_source_lag,
        first_disconnected_distance=local_max_source_lag + 2,
        influence_atol=_INFLUENCE_ATOL,
        gradient_objective="fixed Rademacher VJP of final-position logits",
        local_full_config_difference=config_diff,
        local_full_state_dict_identical=control_state_identical,
        hybrid_minus_local_parameter_count=hybrid_delta,
        hybrid_parameter_delta_fraction=(
            hybrid_delta / variants["hybrid"].parameter_count
        ),
        variants=variants,
        claim_boundary=(
            "This randomly initialized nano-model check combines discrete token "
            "perturbations with a fixed embedding-output VJP. The VJP holds hash routing "
            "and sparse top-k choices fixed and witnesses one local derivative; neither "
            "signal is evidence of learned retrieval, "
            "language quality, efficiency, optimized-kernel behavior, pretrained-checkpoint "
            "behavior, or general long-context performance."
        ),
        passed=passed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check native DeepSeek-V4 attention-path reachability with fixed-seed "
            "source-token perturbations."
        )
    )
    parser.add_argument("--model-seed", type=int, default=_DEFAULT_MODEL_SEED)
    parser.add_argument("--input-seed", type=int, default=_DEFAULT_INPUT_SEED)
    parser.add_argument("--gradient-seed", type=int, default=_DEFAULT_GRADIENT_SEED)
    parser.add_argument("--probe-count", type=int, default=_DEFAULT_PROBE_COUNT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Write the canonical JSON report.")
    parser.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return parser


def _json_payload(report: AttentionReachReport) -> str:
    return json.dumps(
        report.to_dict(),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def _print_human(report: AttentionReachReport) -> None:
    print("nano-deepseek-v4 attention-path reachability")
    print(
        "local boundary: "
        f"max source lag {report.local_max_source_lag} "
        f"(distance {report.local_max_source_lag + 1} reachable; "
        f"distance {report.first_disconnected_distance} disconnected)"
    )
    for name in ("hybrid", "local", "full_window"):
        variant = report.variants[name]
        perturbation_counts = ", ".join(
            f"d{item.distance}={item.materially_changed_probe_count}/{item.probe_count}"
            for item in variant.perturbation_observations
        )
        gradient_counts = ", ".join(
            f"d{item.distance}={item.nonzero_probe_count}/{item.probe_count}"
            for item in variant.gradient_observations
        )
        print(
            f"{name} perturbation: {perturbation_counts} | "
            f"{'PASS' if variant.perturbation_passed else 'FAIL'}"
        )
        print(
            f"{name} embedding VJP: {gradient_counts} | "
            f"{'PASS' if variant.gradient_passed else 'FAIL'}"
        )
    print(f"parameter delta (hybrid-local): {report.hybrid_minus_local_parameter_count:+d}")
    print("scope: structural influence only; no learned-quality claim")
    print(f"overall: {'PASS' if report.passed else 'FAIL'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_attention_reach(
            model_seed=args.model_seed,
            input_seed=args.input_seed,
            gradient_seed=args.gradient_seed,
            probe_count=args.probe_count,
            device=args.device,
        )
        payload = _json_payload(report)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
    except (RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))

    if args.json:
        print(payload, end="")
    else:
        _print_human(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
