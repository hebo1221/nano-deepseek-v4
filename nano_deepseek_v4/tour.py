from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import __version__
from .config import DeepSeekV4Config
from .modeling import DeepSeekV4ForCausalLM

_WALKTHROUGH_PATH = Path("docs/guides/modeling-walkthrough.md")


def _walkthrough_url(*, source_root: Path | None = None) -> str:
    root = Path(__file__).resolve().parent.parent if source_root is None else source_root
    ref = "main" if (root / _WALKTHROUGH_PATH).is_file() else f"v{__version__}"
    return (
        "https://github.com/hebo1221/nano-deepseek-v4/blob/"
        f"{ref}/{_WALKTHROUGH_PATH.as_posix()}"
    )


@dataclass(frozen=True)
class LayerTour:
    """Observed shapes and configured choices for one backbone layer."""

    layer_index: int
    attention_kind: str
    mlp_kind: str
    stream_input_shape: tuple[int, ...]
    attention_input_shape: tuple[int, ...]
    attention_output_shape: tuple[int, ...]
    compression_rate: int | None
    compressed_slot_count: int
    selected_slots_per_query: tuple[int, int] | None
    moe_output_shape: tuple[int, ...]
    router_logits_shape: tuple[int, ...]
    stream_output_shape: tuple[int, ...]
    routing_summary: str


@dataclass(frozen=True)
class MTPTour:
    """Observed shapes for one Multi-Token Prediction depth."""

    depth: int
    attention_kind: str
    previous_streams_shape: tuple[int, ...]
    future_embeddings_shape: tuple[int, ...]
    output_streams_shape: tuple[int, ...]
    hidden_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]


@dataclass(frozen=True)
class TourReport:
    """Structured result of the fixed, executable architecture walkthrough."""

    schema_version: int
    package_version: str
    fixture: str
    device: str
    seed: int
    config_sha256: str
    parameter_count: int
    input_ids: tuple[int, ...]
    input_shape: tuple[int, ...]
    embedding_shape: tuple[int, ...]
    initial_streams_shape: tuple[int, ...]
    layers: tuple[LayerTour, ...]
    final_collapsed_shape: tuple[int, ...]
    final_hidden_shape: tuple[int, ...]
    logits_shape: tuple[int, ...]
    mtp: tuple[MTPTour, ...]
    prefix_cache_tokens: int
    final_cache_tokens: int
    cache_max_abs_error: float
    cache_matches: bool
    claim_boundary: str
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable representation."""

        return asdict(self)


def _tour_config() -> DeepSeekV4Config:
    """Return a tiny fixture that activates every backbone attention family."""

    return DeepSeekV4Config(
        sliding_window=4,
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        index_topk=2,
    )


def _config_sha256(config: DeepSeekV4Config) -> str:
    payload = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _shape(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(value.shape)


def _captured_shape(record: dict[str, Any], key: str) -> tuple[int, ...]:
    value = record.get(key)
    if not isinstance(value, tuple) or not all(isinstance(item, int) for item in value):
        raise RuntimeError(f"architecture tour did not capture {key}")
    return value


def _register_trace_hooks(
    model: DeepSeekV4ForCausalLM,
) -> tuple[list[Any], dict[str, Any]]:
    captures: dict[str, Any] = {
        "embedding_shape": None,
        "final_collapsed_shape": None,
        "final_hidden_shape": None,
        "layers": [dict() for _ in model.model.layers],
        "mtp": [dict() for _ in model.mtp_modules],
    }
    handles: list[Any] = []

    def capture_embedding(
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs
        if isinstance(output, torch.Tensor) and captures["embedding_shape"] is None:
            captures["embedding_shape"] = _shape(output)

    handles.append(model.model.embed_tokens.register_forward_hook(capture_embedding))

    def capture_final_collapse(
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs
        if isinstance(output, torch.Tensor):
            captures["final_collapsed_shape"] = _shape(output)

    def capture_final_hidden(
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs
        if isinstance(output, torch.Tensor):
            captures["final_hidden_shape"] = _shape(output)

    handles.append(model.model.hc_head.register_forward_hook(capture_final_collapse))
    handles.append(model.model.norm.register_forward_hook(capture_final_hidden))

    for layer_index, layer in enumerate(model.model.layers):
        record = captures["layers"][layer_index]

        def capture_layer(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, Any] = record,
        ) -> None:
            del module
            streams = inputs[0]
            output_streams, router_logits = output
            target["stream_input_shape"] = _shape(streams)
            target["stream_output_shape"] = _shape(output_streams)
            target["router_logits_shape"] = _shape(router_logits)

        def capture_attention(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, Any] = record,
        ) -> None:
            del module
            target["attention_input_shape"] = _shape(inputs[0])
            target["attention_output_shape"] = _shape(output)

        def capture_moe(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, Any] = record,
        ) -> None:
            del module, inputs
            moe_output, _ = output
            target["moe_output_shape"] = _shape(moe_output)

        handles.append(layer.register_forward_hook(capture_layer))
        handles.append(layer.self_attn.register_forward_hook(capture_attention))
        handles.append(layer.moe.register_forward_hook(capture_moe))

        compressor = layer.self_attn.csa or layer.self_attn.hca
        if compressor is not None:

            def capture_compressor(
                module: nn.Module,
                inputs: tuple[Any, ...],
                output: Any,
                *,
                target: dict[str, Any] = record,
            ) -> None:
                del inputs
                compressed = output[0]
                target["compression_rate"] = int(module.rate)
                target["compressed_slot_count"] = int(compressed.shape[2])
                if len(output) == 3 and output[2] is not None:
                    selected = output[2].sum(dim=-1)
                    target["selected_slots_per_query"] = (
                        int(selected.min().item()),
                        int(selected.max().item()),
                    )

            handles.append(compressor.register_forward_hook(capture_compressor))

    for depth, mtp_module in enumerate(model.mtp_modules, start=1):
        record = captures["mtp"][depth - 1]

        def capture_mtp(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
            *,
            target: dict[str, Any] = record,
        ) -> None:
            del module
            output_streams, hidden = output
            target["previous_streams_shape"] = _shape(inputs[0])
            target["future_embeddings_shape"] = _shape(inputs[1])
            target["output_streams_shape"] = _shape(output_streams)
            target["hidden_shape"] = _shape(hidden)

        handles.append(mtp_module.register_forward_hook(capture_mtp))

    return handles, captures


def run_tour(*, seed: int = 0) -> TourReport:
    """Execute a tiny CPU model and expose the architecture's real tensor path."""

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        config = _tour_config()
        model = DeepSeekV4ForCausalLM(config).to(device="cpu").eval()
        input_ids = torch.arange(8, dtype=torch.long, device="cpu").unsqueeze(0)
        handles, captures = _register_trace_hooks(model)
        try:
            with torch.inference_mode():
                traced = model(
                    input_ids,
                    labels=input_ids,
                    output_router_logits=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        with torch.inference_mode():
            prefix = model(input_ids[:, :5], use_cache=True)
            if prefix.past_key_values is None:
                raise RuntimeError("architecture tour expected an inference cache")
            prefix_cache_tokens = prefix.past_key_values.get_seq_length()
            suffix = model(
                input_ids[:, 5:],
                past_key_values=prefix.past_key_values,
                use_cache=True,
            )
            if suffix.past_key_values is None:
                raise RuntimeError("architecture tour expected an updated cache")
            cached_logits = torch.cat([prefix.logits, suffix.logits], dim=1)
            cache_max_abs_error = float(
                (cached_logits - traced.logits).abs().max().item()
            )
            cache_matches = bool(
                torch.allclose(cached_logits, traced.logits, atol=1e-4, rtol=1e-4)
            )

        layer_tours: list[LayerTour] = []
        if config.layer_types is None or config.mlp_layer_types is None:
            raise RuntimeError("architecture tour schedules were not initialized")
        for layer_index, (attention_kind, mlp_kind) in enumerate(
            zip(config.layer_types, config.mlp_layer_types, strict=True)
        ):
            record = captures["layers"][layer_index]
            routing_summary = (
                f"token-id hash chooses {config.num_experts_per_tok} of "
                f"{config.n_routed_experts} routed experts"
                if mlp_kind == "hash_moe"
                else f"learned top-k chooses {config.num_experts_per_tok} of "
                f"{config.n_routed_experts} routed experts"
            )
            selected = record.get("selected_slots_per_query")
            if selected is not None:
                selected = tuple(selected)
            layer_tours.append(
                LayerTour(
                    layer_index=layer_index,
                    attention_kind=attention_kind,
                    mlp_kind=mlp_kind,
                    stream_input_shape=_captured_shape(record, "stream_input_shape"),
                    attention_input_shape=_captured_shape(record, "attention_input_shape"),
                    attention_output_shape=_captured_shape(record, "attention_output_shape"),
                    compression_rate=record.get("compression_rate"),
                    compressed_slot_count=int(record.get("compressed_slot_count", 0)),
                    selected_slots_per_query=selected,
                    moe_output_shape=_captured_shape(record, "moe_output_shape"),
                    router_logits_shape=_captured_shape(record, "router_logits_shape"),
                    stream_output_shape=_captured_shape(record, "stream_output_shape"),
                    routing_summary=routing_summary,
                )
            )

        mtp_tours: list[MTPTour] = []
        mtp_logits = traced.mtp_logits or []
        for depth, (record, logits) in enumerate(
            zip(captures["mtp"], mtp_logits, strict=True),
            start=1,
        ):
            if config.mtp_layer_types is None:
                raise RuntimeError("architecture tour MTP schedule was not initialized")
            mtp_tours.append(
                MTPTour(
                    depth=depth,
                    attention_kind=config.mtp_layer_types[depth - 1],
                    previous_streams_shape=_captured_shape(
                        record, "previous_streams_shape"
                    ),
                    future_embeddings_shape=_captured_shape(
                        record, "future_embeddings_shape"
                    ),
                    output_streams_shape=_captured_shape(
                        record, "output_streams_shape"
                    ),
                    hidden_shape=_captured_shape(record, "hidden_shape"),
                    logits_shape=_shape(logits),
                )
            )

        initial_streams_shape = layer_tours[0].stream_input_shape
        embedding_shape = _captured_shape(captures, "embedding_shape")
        final_collapsed_shape = _captured_shape(captures, "final_collapsed_shape")
        final_hidden_shape = _captured_shape(captures, "final_hidden_shape")

        parameters = sum(parameter.numel() for parameter in model.parameters())

    return TourReport(
        schema_version=1,
        package_version=__version__,
        fixture="tiny-architecture-tour-v1",
        device="cpu",
        seed=seed,
        config_sha256=_config_sha256(config),
        parameter_count=parameters,
        input_ids=tuple(int(token) for token in input_ids[0].tolist()),
        input_shape=_shape(input_ids),
        embedding_shape=embedding_shape,
        initial_streams_shape=initial_streams_shape,
        layers=tuple(layer_tours),
        final_collapsed_shape=final_collapsed_shape,
        final_hidden_shape=final_hidden_shape,
        logits_shape=_shape(traced.logits),
        mtp=tuple(mtp_tours),
        prefix_cache_tokens=prefix_cache_tokens,
        final_cache_tokens=suffix.past_key_values.get_seq_length(),
        cache_max_abs_error=cache_max_abs_error,
        cache_matches=cache_matches,
        claim_boundary=(
            "This fixed CPU walkthrough explains observed tensor shapes and configured "
            "routing in a deliberately tiny educational fixture. It is not evidence of "
            "official-checkpoint execution, model quality, or accelerator performance."
        ),
        passed=cache_matches and len(layer_tours) == 4 and len(mtp_tours) == 1,
    )


def _short_attention(kind: str) -> str:
    return kind.removesuffix("_attention")


def _print_human(report: TourReport) -> None:
    print("nano-deepseek-v4 architecture tour")
    print(
        f"fixture: {report.fixture} · {report.parameter_count:,} parameters · "
        f"{len(report.input_ids)} CPU tokens"
    )
    print(f"tokens: {report.input_shape} -> embeddings: {report.embedding_shape}")
    print(
        f"mHC expands one hidden state into {report.initial_streams_shape[2]} streams: "
        f"{report.initial_streams_shape}"
    )
    print("backbone:")
    for layer in report.layers:
        compression = "local KV only"
        if layer.compression_rate is not None:
            compression = (
                f"compress {layer.compression_rate}:1 -> "
                f"{layer.compressed_slot_count} slots"
            )
            if layer.selected_slots_per_query is not None:
                low, high = layer.selected_slots_per_query
                compression += f", CSA selects {low}..{high} causal slots/query"
        print(
            f"  L{layer.layer_index}: {_short_attention(layer.attention_kind)} + "
            f"{layer.mlp_kind}"
        )
        print(
            f"      streams {layer.stream_input_shape} -> attention "
            f"{layer.attention_input_shape} -> {layer.attention_output_shape}"
        )
        print(f"      memory: {compression}")
        print(
            f"      routing: {layer.routing_summary}; logits "
            f"{layer.router_logits_shape}"
        )
    print(
        f"head: streams -> {report.final_collapsed_shape} -> normalized hidden "
        f"{report.final_hidden_shape} -> logits {report.logits_shape}"
    )
    for mtp in report.mtp:
        print(
            f"MTP depth {mtp.depth}: previous streams {mtp.previous_streams_shape} + "
            f"future embeddings {mtp.future_embeddings_shape} -> logits {mtp.logits_shape}"
        )
    print(
        f"cache: {report.prefix_cache_tokens} -> {report.final_cache_tokens} tokens; "
        f"cached/full match: {report.cache_matches}"
    )
    print(f"read next: {_walkthrough_url()}")
    print(f"scope: {report.claim_boundary}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Trace the real tensor shapes through a tiny CPU DeepSeek-V4 model."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="model initialization seed")
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    report = run_tour(seed=args.seed)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    else:
        _print_human(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
