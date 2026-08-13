from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .checkpoint import (
    OfficialCheckpointNamespaceReport,
    OfficialCheckpointSnapshotReport,
    OfficialHubCheckpointInspection,
    PretrainedBundleReport,
    estimate_deepseek_v4_parameter_counts,
    inspect_deepseek_checkpoint_namespace,
    inspect_deepseek_hub_checkpoint_namespace,
    verify_deepseek_checkpoint_snapshot,
    verify_deepseek_v4_pretrained_bundle,
)
from .config import DeepSeekV4Config


@dataclass(frozen=True)
class ArchitectureReport:
    """Allocation-free summary of a native or official DeepSeek-V4 config."""

    schema_version: int
    source: str
    config_sha256: str
    total_parameter_count: int
    model_parameter_count: int
    non_parameter_routing_state_count: int
    activated_parameter_count: int
    activated_fraction: float
    hidden_size: int
    num_hidden_layers: int
    num_nextn_predict_layers: int
    auxiliary_kind: str
    num_mtp_modules: int
    dspark_stage_count: int
    dspark_block_size: int
    dspark_target_layer_ids: list[int]
    dspark_markov_rank: int | None
    max_position_embeddings: int
    rope_theta: float
    compressed_rope_theta: float
    rope_scaling: dict[str, Any] | None
    routed_expert_count: int
    experts_per_token: int
    attention_layer_counts: dict[str, int]
    mtp_attention_layer_counts: dict[str, int]
    dspark_attention_layer_counts: dict[str, int]
    mlp_layer_counts: dict[str, int]
    speculative_parameter_count: int
    speculative_non_parameter_routing_state_count: int
    speculative_activated_parameter_count: int
    runtime_load_supported: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _config_sha256(config: DeepSeekV4Config) -> str:
    canonical = json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def inspect_architecture(
    config: DeepSeekV4Config,
    *,
    source: str = "native config",
) -> ArchitectureReport:
    """Summarize schedules and official-schema parameter counts without a model."""

    counts = estimate_deepseek_v4_parameter_counts(config)
    total = counts["total_parameters"]
    activated = counts["activated_parameters"]
    if config.has_dspark:
        auxiliary_kind = "dspark"
    elif config.num_nextn_predict_layers > 0:
        auxiliary_kind = "mtp"
    else:
        auxiliary_kind = "none"
    return ArchitectureReport(
        schema_version=3,
        source=source,
        config_sha256=_config_sha256(config),
        total_parameter_count=total,
        model_parameter_count=counts["model_parameters"],
        non_parameter_routing_state_count=counts["non_parameter_routing_state"],
        activated_parameter_count=activated,
        activated_fraction=activated / total,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_nextn_predict_layers=config.num_nextn_predict_layers,
        auxiliary_kind=auxiliary_kind,
        num_mtp_modules=(
            0 if config.has_dspark else config.num_nextn_predict_layers
        ),
        dspark_stage_count=config.dspark_stage_count,
        dspark_block_size=config.dspark_block_size,
        dspark_target_layer_ids=list(config.dspark_target_layer_ids),
        dspark_markov_rank=(
            config.dspark_markov_rank if config.has_dspark else None
        ),
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=float(config.rope_theta),
        compressed_rope_theta=float(config.compress_rope_theta),
        rope_scaling=(
            dict(config.rope_scaling) if config.rope_scaling is not None else None
        ),
        routed_expert_count=config.n_routed_experts,
        experts_per_token=config.num_experts_per_tok,
        attention_layer_counts=dict(Counter(config.layer_types or [])),
        mtp_attention_layer_counts=(
            {}
            if config.has_dspark
            else dict(Counter(config.mtp_layer_types or []))
        ),
        dspark_attention_layer_counts=(
            dict(Counter(config.dspark_layer_types or []))
            if config.has_dspark
            else {}
        ),
        mlp_layer_counts=dict(Counter(config.mlp_layer_types or [])),
        speculative_parameter_count=counts["speculative_parameters"],
        speculative_non_parameter_routing_state_count=counts[
            "speculative_non_parameter_routing_state"
        ],
        speculative_activated_parameter_count=counts[
            "speculative_activated_parameters"
        ],
        runtime_load_supported=not config.has_dspark,
    )


def _format_count(value: int) -> str:
    if value >= 10**12:
        return f"{value / 10**12:.3f}T"
    if value >= 10**9:
        return f"{value / 10**9:.3f}B"
    if value >= 10**6:
        return f"{value / 10**6:.3f}M"
    if value >= 10**3:
        return f"{value / 10**3:.3f}K"
    return f"{value:,}"


def _format_schedule(counts: dict[str, int]) -> str:
    return " | ".join(f"{name}={count}" for name, count in counts.items())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect DeepSeek-V4 configurations and checkpoint headers without "
            "allocating a model."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--preset",
        choices=("tiny", "flash", "flash-0731", "pro"),
    )
    source.add_argument("--config", type=Path, help="Native nano-deepseek-v4 config JSON.")
    source.add_argument("--official-config", type=Path, help="Official DeepSeek config JSON.")
    source.add_argument(
        "--bundle",
        type=Path,
        help="Checksummed native model bundle to verify and inspect.",
    )
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="Official checkpoint directory to verify and inspect without loading payloads.",
    )
    source.add_argument(
        "--hf-repo",
        help="Hugging Face model repo to inspect without caching weight files.",
    )
    parser.add_argument(
        "--namespace",
        help="With --checkpoint or --hf-repo, audit a namespace such as mtp or mtp.0.",
    )
    parser.add_argument(
        "--revision",
        help="Branch, tag, or commit for --hf-repo; resolved to an immutable commit SHA.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory for config and index documents.",
    )
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    return parser


def _resolve_config(
    args: argparse.Namespace,
) -> tuple[
    DeepSeekV4Config,
    str,
    PretrainedBundleReport | None,
    OfficialCheckpointSnapshotReport | None,
    OfficialHubCheckpointInspection | None,
    OfficialCheckpointNamespaceReport | None,
]:
    if args.namespace is not None and args.checkpoint is None and args.hf_repo is None:
        raise ValueError("--namespace requires --checkpoint or --hf-repo.")
    if args.hf_repo is not None and args.namespace is None:
        raise ValueError("--hf-repo requires --namespace.")
    if args.hf_repo is not None and args.revision is None:
        raise ValueError("--hf-repo requires --revision.")
    if args.hf_repo is None and args.revision is not None:
        raise ValueError("--revision requires --hf-repo.")
    if args.hf_repo is None and args.hf_cache_dir is not None:
        raise ValueError("--hf-cache-dir requires --hf-repo.")
    if args.config is not None:
        return (
            DeepSeekV4Config.from_json_file(args.config),
            str(args.config),
            None,
            None,
            None,
            None,
        )
    if args.official_config is not None:
        return (
            DeepSeekV4Config.from_official_json(args.official_config),
            str(args.official_config),
            None,
            None,
            None,
            None,
        )
    if args.bundle is not None:
        bundle_report = verify_deepseek_v4_pretrained_bundle(args.bundle)
        if not bundle_report.is_complete:
            raise ValueError(
                "bundle verification failed: " + "; ".join(bundle_report.errors)
            )
        return (
            DeepSeekV4Config.from_json_file(args.bundle / "config.json"),
            f"bundle:{args.bundle}",
            bundle_report,
            None,
            None,
            None,
        )
    if args.checkpoint is not None:
        checkpoint_report = verify_deepseek_checkpoint_snapshot(args.checkpoint)
        if not checkpoint_report.is_complete:
            issue_count = sum(
                len(items)
                for items in (
                    checkpoint_report.missing_shards,
                    checkpoint_report.missing_expected_keys,
                    checkpoint_report.index_metadata_errors,
                    checkpoint_report.dtype_metadata_errors,
                    checkpoint_report.missing_keys_in_shards,
                    checkpoint_report.unexpected_keys_in_shards,
                    checkpoint_report.shape_mismatches,
                    checkpoint_report.unchecked_shape_keys,
                    checkpoint_report.coverage.unrecognized_keys,
                )
            )
            raise ValueError(
                f"checkpoint verification failed with {issue_count} issue(s)."
            )
        checkpoint_root = (
            args.checkpoint if args.checkpoint.is_dir() else args.checkpoint.parent
        )
        namespace_report = (
            inspect_deepseek_checkpoint_namespace(args.checkpoint, args.namespace)
            if args.namespace is not None
            else None
        )
        if namespace_report is not None and not namespace_report.is_complete:
            raise ValueError(
                f"checkpoint namespace {args.namespace!r} failed verification."
            )
        return (
            DeepSeekV4Config.from_official_json(checkpoint_root / "config.json"),
            f"checkpoint:{args.checkpoint}",
            None,
            checkpoint_report,
            None,
            namespace_report,
        )
    if args.hf_repo is not None:
        hub_report = inspect_deepseek_hub_checkpoint_namespace(
            args.hf_repo,
            args.revision,
            args.namespace,
            cache_dir=args.hf_cache_dir,
        )
        if not hub_report.is_complete:
            raise ValueError(
                f"Hub checkpoint namespace {args.namespace!r} failed verification."
            )
        return (
            hub_report.config,
            f"hf:{args.hf_repo}@{hub_report.resolved_revision}",
            None,
            None,
            hub_report,
            hub_report.namespace,
        )
    preset = args.preset or "tiny"
    if preset == "flash":
        config = DeepSeekV4Config.flash()
    elif preset == "flash-0731":
        config = DeepSeekV4Config.flash_0731()
    elif preset == "pro":
        config = DeepSeekV4Config.pro()
    else:
        config = DeepSeekV4Config()
    return config, f"preset:{preset}", None, None, None, None


def _checkpoint_summary(
    report: OfficialCheckpointSnapshotReport,
) -> dict[str, Any]:
    return {
        "is_complete": report.is_complete,
        "index_path": report.index_path,
        "total_keys": report.total_keys,
        "total_shards": report.total_shards,
        "total_size_bytes": report.total_size_bytes,
        "total_tensor_bytes": report.total_tensor_bytes,
        "dtype_counts": report.dtype_counts,
        "quantized_tensor_count": report.quantized_tensor_count,
        "scale_tensor_count": report.scale_tensor_count,
        "unrecognized_key_count": len(report.coverage.unrecognized_keys),
        "shape_mismatch_count": len(report.shape_mismatches),
        "unchecked_shape_key_count": len(report.unchecked_shape_keys),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        (
            config,
            source,
            bundle_report,
            checkpoint_report,
            hub_report,
            namespace_report,
        ) = _resolve_config(args)
        report = inspect_architecture(config, source=source)
    except (FileNotFoundError, ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    if args.json:
        payload = report.to_dict()
        if bundle_report is not None:
            payload["bundle"] = bundle_report.to_dict()
        if checkpoint_report is not None:
            payload["checkpoint"] = _checkpoint_summary(checkpoint_report)
        if hub_report is not None:
            payload["hub_checkpoint"] = hub_report.to_dict()
        if namespace_report is not None:
            payload["namespace"] = namespace_report.to_dict()
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("nano-deepseek-v4 architecture")
        print(f"source: {report.source}")
        print(
            f"logical parameters: {report.total_parameter_count:,} "
            f"({_format_count(report.total_parameter_count)})"
        )
        print(f"nn.Parameter: {report.model_parameter_count:,}")
        print(
            "non-parameter routing state: "
            f"{report.non_parameter_routing_state_count:,}"
        )
        print(
            f"activated: {report.activated_parameter_count:,} "
            f"({_format_count(report.activated_parameter_count)}, "
            f"{report.activated_fraction:.2%})"
        )
        if report.auxiliary_kind == "dspark":
            print(
                f"layers: {report.num_hidden_layers} backbone + "
                f"{report.dspark_stage_count} DSpark stages"
            )
            print(
                "config num_nextn_predict_layers: "
                f"{report.num_nextn_predict_layers}"
            )
        else:
            print(
                f"layers: {report.num_hidden_layers} backbone + "
                f"{report.num_mtp_modules} MTP"
            )
        print(f"attention: {_format_schedule(report.attention_layer_counts)}")
        if report.auxiliary_kind == "dspark":
            print(
                "DSpark attention: "
                f"{_format_schedule(report.dspark_attention_layer_counts)}"
            )
            print(
                f"DSpark: block size={report.dspark_block_size} | "
                "targets="
                + ",".join(str(index) for index in report.dspark_target_layer_ids)
                + f" | Markov rank={report.dspark_markov_rank}"
            )
            print(
                "speculative parameters: "
                f"{report.speculative_parameter_count:,} "
                f"({_format_count(report.speculative_parameter_count)})"
            )
            print(
                "speculative activated: "
                f"{report.speculative_activated_parameter_count:,} "
                f"({_format_count(report.speculative_activated_parameter_count)})"
            )
            print("runtime load supported: no")
        else:
            print(
                "MTP attention: "
                f"{_format_schedule(report.mtp_attention_layer_counts)}"
            )
        print(f"mlp: {_format_schedule(report.mlp_layer_counts)}")
        print(f"context: {report.max_position_embeddings:,} tokens")
        if report.rope_scaling is None:
            print(
                f"rotary: main theta={report.rope_theta:g} | "
                f"compressed theta={report.compressed_rope_theta:g}"
            )
        else:
            print(
                f"rotary: main theta={report.rope_theta:g} | "
                f"compressed theta={report.compressed_rope_theta:g} + "
                f"{report.rope_scaling['type']} factor={report.rope_scaling['factor']:g} "
                f"from {report.rope_scaling['original_max_position_embeddings']:,} tokens"
            )
        print(f"config sha256: {report.config_sha256}")
        if bundle_report is not None:
            print(
                f"bundle: verified {bundle_report.shard_count} shard(s), "
                f"{bundle_report.tensor_count:,} tensors, "
                f"{bundle_report.tensor_bytes:,} payload bytes"
            )
            if bundle_report.generation_ready:
                print(
                    "tokenizer: checksum-bound byte-v1 "
                    f"({bundle_report.tokenizer_sha256})"
                )
            else:
                print("tokenizer: not bundled (model loading only)")
        if checkpoint_report is not None:
            print(
                f"checkpoint: verified {checkpoint_report.total_shards} shard(s), "
                f"{checkpoint_report.total_keys:,} tensors, "
                f"{checkpoint_report.total_tensor_bytes:,} payload bytes"
            )
        if hub_report is not None:
            print(
                f"Hub checkpoint: metadata-only inspection of "
                f"{hub_report.inspected_shard_count}/{hub_report.total_shard_count} shard header(s)"
            )
            print(
                f"Hub revision: {hub_report.requested_revision} -> "
                f"{hub_report.resolved_revision}"
            )
            print(f"huggingface_hub: {hub_report.huggingface_hub_version}")
            print(
                f"metadata documents: {hub_report.metadata_document_bytes:,} bytes; "
                "full snapshot preflight: not performed"
            )
        if namespace_report is not None:
            print(
                f"namespace {namespace_report.namespace}: "
                f"{namespace_report.inspected_tensor_count:,} tensors in "
                f"{len(namespace_report.shard_files)} shard(s), "
                f"{namespace_report.stored_tensor_bytes:,} bytes"
            )
            print(f"namespace inventory sha256: {namespace_report.inventory_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
