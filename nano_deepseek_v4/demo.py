from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from . import __version__
from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM


@dataclass(frozen=True)
class DemoReport:
    """Structured receipt for the tiny no-download CPU smoke test."""

    schema_version: int
    package_version: str
    python_version: str
    torch_version: str
    device: str
    seed: int
    config_sha256: str
    parameter_count: int
    attention_schedule: tuple[str, ...]
    mlp_schedule: tuple[str, ...]
    input_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    prefix_cache_tokens: int
    final_cache_tokens: int
    router_layer_count: int
    cache_max_abs_error: float
    cache_matches: bool
    claim_boundary: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable representation."""

        return asdict(self)


def _config_sha256(config: DeepSeekV4Config) -> str:
    canonical = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def run_demo(*, seed: int = 0) -> DemoReport:
    """Run the tiny CPU architecture/cache smoke test without changing caller RNG."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = DeepSeekV4Config()
        model = DeepSeekV4ForCausalLM(config).to(device="cpu").eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 8), device="cpu")

        with torch.inference_mode():
            full = model(input_ids, output_router_logits=True)
            prefix = model(input_ids[:, :5], use_cache=True)
            if prefix.past_key_values is None:
                raise RuntimeError("demo expected an inference cache")
            prefix_tokens = prefix.past_key_values.get_seq_length()
            suffix = model(
                input_ids[:, 5:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
            if suffix.past_key_values is None:
                raise RuntimeError("demo expected an updated inference cache")
            cached_logits = torch.cat([prefix.logits, suffix.logits], dim=1)
            cache_max_abs_error = float(
                (cached_logits - full.logits).abs().max().item()
            )
            cache_matches = bool(
                torch.allclose(cached_logits, full.logits, atol=1e-4, rtol=1e-4)
            )

        parameters = sum(parameter.numel() for parameter in model.parameters())
        final_cache_tokens = suffix.past_key_values.get_seq_length()

    return DemoReport(
        schema_version=1,
        package_version=__version__,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        device="cpu",
        seed=seed,
        config_sha256=_config_sha256(config),
        parameter_count=parameters,
        attention_schedule=tuple(config.layer_types or ()),
        mlp_schedule=tuple(config.mlp_layer_types or ()),
        input_shape=tuple(input_ids.shape),
        logits_shape=tuple(full.logits.shape),
        prefix_cache_tokens=prefix_tokens,
        final_cache_tokens=final_cache_tokens,
        router_layer_count=len(full.router_logits or []),
        cache_max_abs_error=cache_max_abs_error,
        cache_matches=cache_matches,
        claim_boundary=(
            "This no-download CPU smoke test covers only the tiny native model's full "
            "forward and cache equivalence; it does not establish official-checkpoint "
            "parity, training quality, or accelerator performance."
        ),
        passed=cache_matches,
    )


def main(argv: Sequence[str] = ()) -> int:
    parser = argparse.ArgumentParser(
        description="Run the no-download tiny CPU architecture and cache smoke test.",
    )
    parser.parse_args(argv)
    report = run_demo()
    schedule = " -> ".join(
        layer.replace("_attention", "") for layer in report.attention_schedule
    )
    mlp_schedule = " -> ".join(report.mlp_schedule)

    print(f"nano-deepseek-v4 {report.package_version}")
    print(f"parameters: {report.parameter_count:,}")
    print(f"attention: {schedule}")
    print(f"mlp: {mlp_schedule}")
    print(f"logits: {report.logits_shape}")
    print(f"cache tokens: {report.prefix_cache_tokens} -> {report.final_cache_tokens}")
    print(f"cached/full match: {report.cache_matches}")
    print(f"router layers: {report.router_layer_count}")
    return 0 if report.passed else 1


def cli_main(argv: Sequence[str] | None = None) -> int:
    """Run the demo as a command while keeping ``main()`` embedding-safe."""

    return main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    raise SystemExit(cli_main())
