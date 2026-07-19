from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import adaptive_v4_execution_environment as execution_environment
import p2_direct_attestation as attestation
import torch
import torch.nn.functional as F

from nano_deepseek_v4 import (
    AssociativeRecallConfig,
    CSAProbeObjective,
    CSASelectionProbe,
    DeepSeekV4Config,
    DeepSeekV4ForCausalLM,
    generate_associative_recall_batch,
    generate_associative_recall_training_batch,
)

SCALE_OVERRIDES = {
    "s55": {
        "hidden_size": 384,
        "moe_intermediate_size": 512,
        "num_hidden_layers": 8,
    },
    "s151": {
        "hidden_size": 512,
        "moe_intermediate_size": 768,
        "num_hidden_layers": 12,
    },
}
SEQUENCE_LENGTHS = (48, 64, 80)
TRAIN_SEQUENCE_LENGTHS = (64, 80)
DIRECT_TRAINING_EXPERIMENT_ID = "p2-post-rank-direct-training-v1"
DIRECT_SUMMARY_SCHEMA_VERSION = 3
DIRECT_CHECKPOINT_SCHEMA_VERSION = 4
DIRECT_TRANSCRIPT_SCHEME = "sha256-ordered-training-step-chain-v1"
DIRECT_SUMMARY_ATTESTATION_PURPOSE = "p2-direct-training-summary-v1"
DIRECT_TRAINING_HYPERPARAMETERS: dict[str, Any] = {
    "steps": 1_000,
    "minimum_steps": 1_000,
    "batch_size": 16,
    "num_queries": 4,
    "learning_rate": 1e-3,
    "weight_decay": 0.01,
    "ranking_loss_weight": 1.0,
    "read_ranking_loss_weight": 1.0,
    "value_loss_weight": 1.0,
    "gradient_clip_norm": 1.0,
    "training_topk": 64,
    "eval_every": 50,
    "eval_batches": 2,
    "eval_batch_size": 32,
    "early_stopping_enabled": False,
    "optimizer": {
        "name": "AdamW",
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
        "fused": True,
    },
    "autocast_dtype": "bfloat16",
    "device_type": "cuda",
    "sequence_lengths": list(SEQUENCE_LENGTHS),
    "training_sequence_lengths": list(TRAIN_SEQUENCE_LENGTHS),
    "task_geometry": {
        "vocab_size": 4096,
        "sliding_window": 32,
        "key_count": 64,
        "value_start": 80,
        "value_count": 64,
    },
}


def _source_state() -> dict[str, str | bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def build_config(scale: str) -> DeepSeekV4Config:
    if scale not in SCALE_OVERRIDES:
        raise ValueError(f"unknown Tier-S scale: {scale!r}.")
    values = SCALE_OVERRIDES[scale]
    return DeepSeekV4Config(
        vocab_size=4096,
        hidden_size=values["hidden_size"],
        moe_intermediate_size=values["moe_intermediate_size"],
        num_hidden_layers=values["num_hidden_layers"],
        num_attention_heads=8,
        head_dim=64,
        q_lora_rank=values["hidden_size"] // 2,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hash_layers=3,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        sliding_window=32,
        o_groups=4,
        o_lora_rank=64,
        index_n_heads=8,
        index_head_dim=64,
        index_topk=8,
        num_nextn_predict_layers=1,
    )


def _set_index_topk(model: DeepSeekV4ForCausalLM, topk: int) -> None:
    for layer in model.model.layers:
        if layer.self_attn.csa is not None:
            layer.self_attn.csa.indexer.index_topk = topk


def _answer_logits(
    model: DeepSeekV4ForCausalLM,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    hidden_states, _, _ = model.model(input_ids)
    return model.lm_head(hidden_states[:, -1])


def _query_logits(
    model: DeepSeekV4ForCausalLM,
    input_ids: torch.Tensor,
    query_positions: torch.Tensor,
    csa_probe: CSASelectionProbe | None = None,
) -> torch.Tensor:
    hidden_states, _, _ = model.model(input_ids, csa_probe=csa_probe)
    gather_index = query_positions.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
    query_states = hidden_states.gather(1, gather_index)
    return model.lm_head(query_states)


@torch.no_grad()
def evaluate(
    model: DeepSeekV4ForCausalLM,
    task_config: AssociativeRecallConfig,
    *,
    sequence_lengths: tuple[int, ...],
    batches_per_length: int,
    batch_size: int,
    seed: int,
    topk: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    _set_index_topk(model, topk)
    generator = torch.Generator().manual_seed(seed)
    metrics: dict[str, float] = {}
    total_correct = 0
    total_examples = 0
    for sequence_length in sequence_lengths:
        correct = 0
        examples = 0
        for _ in range(batches_per_length):
            batch = generate_associative_recall_batch(
                task_config,
                batch_size=batch_size,
                sequence_length=sequence_length,
                generator=generator,
                device=device,
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = _answer_logits(model, batch.input_ids)
            correct += int(logits.argmax(dim=-1).eq(batch.targets).sum())
            examples += batch_size
        metrics[f"accuracy_length_{sequence_length}"] = correct / examples
        total_correct += correct
        total_examples += examples
    metrics["mean_accuracy"] = total_correct / total_examples
    return metrics


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def direct_training_hyperparameters() -> dict[str, Any]:
    return json.loads(json.dumps(DIRECT_TRAINING_HYPERPARAMETERS, sort_keys=True))


def canonical_direct_hyperparameters_json() -> str:
    return json.dumps(
        DIRECT_TRAINING_HYPERPARAMETERS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _args_direct_hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    payload = direct_training_hyperparameters()
    payload.update(
        {
            "steps": args.steps,
            "minimum_steps": args.minimum_steps,
            "batch_size": args.batch_size,
            "num_queries": args.num_queries,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ranking_loss_weight": args.ranking_loss_weight,
            "read_ranking_loss_weight": args.read_ranking_loss_weight,
            "value_loss_weight": args.value_loss_weight,
            "gradient_clip_norm": args.gradient_clip_norm,
            "training_topk": args.training_topk,
            "eval_every": args.eval_every,
            "eval_batches": args.eval_batches,
            "eval_batch_size": args.eval_batch_size,
            "early_stopping_enabled": not args.disable_early_stop,
        }
    )
    return payload


def _initial_transcript_root(
    *,
    scale: str,
    seed: int,
    launch_nonce: str,
    manifest_sha256: str,
    trainer_sha256: str,
) -> str:
    return _canonical_json_digest(
        {
            "scheme": DIRECT_TRANSCRIPT_SCHEME,
            "scale": scale,
            "training_seed": seed,
            "launch_nonce": launch_nonce,
            "manifest_sha256": manifest_sha256,
            "trainer_sha256": trainer_sha256,
            "hyperparameters": direct_training_hyperparameters(),
        }
    )


def _extend_transcript_root(previous: str, entry: Mapping[str, object]) -> str:
    if len(previous) != 64:
        raise ValueError("Direct training transcript root is invalid.")
    digest = hashlib.sha256()
    digest.update(bytes.fromhex(previous))
    digest.update(
        json.dumps(
            entry,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    )
    return digest.hexdigest()


def _batch_digest(batch: Any, *, sequence_length: int) -> str:
    return _canonical_json_digest(
        {
            "sequence_length": sequence_length,
            "input_ids": batch.input_ids.detach().to(device="cpu", dtype=torch.long).tolist(),
            "query_positions": batch.query_positions.detach()
            .to(device="cpu", dtype=torch.long)
            .tolist(),
            "evidence_positions": batch.evidence_positions.detach()
            .to(device="cpu", dtype=torch.long)
            .tolist(),
            "targets": batch.targets.detach().to(device="cpu", dtype=torch.long).tolist(),
        }
    )


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _direct_training_contract(
    args: argparse.Namespace,
    *,
    source: Mapping[str, str | bool],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest": {
            "path": str(Path(args.manifest_path).resolve()),
            "sha256": args.manifest_sha256,
            "experiment_id": args.manifest_experiment_id,
            "implementation_digest": args.implementation_digest,
            "implementation_source_commit": args.implementation_source_commit,
            "attestation": attestation.public_manifest_contract(args.attestation_key_id),
        },
        "source_commit": source["commit"],
        "attestation": {
            "scheme": attestation.SCHEME,
            "key_id": args.attestation_key_id,
            "launch_nonce": args.launch_nonce,
        },
        "canonical_trainer_sha256": args.trainer_sha256,
        "hyperparameters": _args_direct_hyperparameters(args),
        "seed_rules": {
            "initialization_seed": "training_seed",
            "data_order_seed": "training_seed+1",
            "training_evaluation_seed": "training_seed+10000",
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_stat = os.fstat(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise ValueError(
                    f"Refusing to overwrite existing training summary: {path}"
                ) from error
            published = attestation.open_regular_nofollow(path)
            try:
                published_stat = os.fstat(published.file_descriptor)
                if (
                    (published_stat.st_dev, published_stat.st_ino)
                    != (temporary_stat.st_dev, temporary_stat.st_ino)
                    or published.bytes != len(encoded)
                    or published.read_bytes() != encoded
                ):
                    raise ValueError("Published training summary bytes changed before validation.")
                published.assert_unchanged()
            finally:
                published.close()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_checkpoint_atomically(
    model: DeepSeekV4ForCausalLM,
    probe_objective: CSAProbeObjective,
    optimizer: torch.optim.Optimizer,
    config: DeepSeekV4Config,
    path: Path,
    *,
    provenance: Mapping[str, object],
) -> dict[str, int | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=".tier-s-",
            suffix=".pt.tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
            probe_state = {
                name: tensor.detach().cpu() for name, tensor in probe_objective.state_dict().items()
            }
            torch.save(
                {
                    "checkpoint_schema_version": DIRECT_CHECKPOINT_SCHEMA_VERSION,
                    "provenance": dict(provenance),
                    "config": asdict(config),
                    "model": state,
                    "probe_objective": probe_state,
                    "optimizer": _cpu_clone(optimizer.state_dict()),
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
            temporary_stat = os.fstat(handle.fileno())
            checkpoint_sha256 = attestation.checksum_fd(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"Refusing to overwrite existing training checkpoint: {path}"
            ) from error
        published = attestation.open_regular_nofollow(path)
        try:
            published_stat = os.fstat(published.file_descriptor)
            if (
                (published_stat.st_dev, published_stat.st_ino)
                != (temporary_stat.st_dev, temporary_stat.st_ino)
                or published.sha256 != checkpoint_sha256
                or published.bytes != temporary_stat.st_size
            ):
                raise ValueError("Published training checkpoint bytes changed before attestation.")
            published.assert_unchanged()
        finally:
            published.close()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": checkpoint_sha256,
        "bytes": temporary_stat.st_size,
    }


def _expected_device_routing_identity(args: argparse.Namespace) -> dict[str, str] | None:
    raw = args.expected_device_routing_identity_json
    if not raw:
        if args.experiment_id == DIRECT_TRAINING_EXPERIMENT_ID:
            raise ValueError("Direct training physical-device guard identity is missing.")
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("Training physical-device guard identity is invalid JSON.") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Training physical-device guard identity is invalid.")
    return execution_environment.validate_device_routing_identity(payload)


def train(args: argparse.Namespace) -> dict:
    initialization_seed = args.seed
    data_order_seed = args.seed + 1
    training_evaluation_seed = args.seed + 10_000
    torch.manual_seed(initialization_seed)
    device, stable_execution_environment = execution_environment.activate_explicit_cuda_device(
        args.device,
        expected_routing_identity=_expected_device_routing_identity(args),
    )
    config = build_config(args.scale)
    source = _source_state()
    trust_root: attestation.TrustRoot | None = None
    if args.experiment_id == DIRECT_TRAINING_EXPERIMENT_ID:
        if args.direct_hyperparameters_json != canonical_direct_hyperparameters_json():
            raise ValueError("Direct training full hyperparameter declaration drifted.")
        if _args_direct_hyperparameters(args) != direct_training_hyperparameters():
            raise ValueError("Direct training hyperparameters drifted from the frozen contract.")
        if args.no_save:
            raise ValueError("Direct training may not disable checkpoint publication.")
        if (
            not isinstance(args.launch_nonce, str)
            or len(args.launch_nonce) != 64
            or any(character not in "0123456789abcdef" for character in args.launch_nonce)
        ):
            raise ValueError("Direct training launch nonce is missing or invalid.")
        for name in ("attestation_key_id", "trainer_sha256"):
            value = getattr(args, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"Direct training {name} binding is missing or invalid.")
        trust_root = attestation.trust_root_from_inherited_environment(
            expected_key_id=args.attestation_key_id
        )
        if source.get("dirty") is not False or source.get("commit") != args.source_commit:
            raise ValueError("Direct training source binding is dirty or mismatched.")
        manifest_path = Path(args.manifest_path).resolve()
        opened_manifest = attestation.open_regular_nofollow(manifest_path)
        try:
            if opened_manifest.sha256 != args.manifest_sha256:
                raise ValueError("Direct training manifest path or digest is invalid.")
            try:
                manifest_payload = json.loads(opened_manifest.read_bytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("Direct training manifest is invalid JSON.") from error
            opened_manifest.assert_unchanged()
        finally:
            opened_manifest.close()
        if (
            not isinstance(manifest_payload, dict)
            or manifest_payload.get("experiment_id") != args.manifest_experiment_id
        ):
            raise ValueError("Direct training manifest experiment binding is invalid.")
        manifest_attestation = manifest_payload.get("attestation")
        if not isinstance(manifest_attestation, dict) or manifest_attestation != (
            attestation.public_manifest_contract(args.attestation_key_id)
        ):
            raise ValueError("Direct training manifest attestation binding is invalid.")
        for name in ("manifest_sha256", "implementation_digest", "implementation_source_commit"):
            value = getattr(args, name)
            allowed_lengths = (40, 64) if name == "implementation_source_commit" else (64,)
            if (
                not isinstance(value, str)
                or len(value) not in allowed_lengths
                or any(character not in "0123456789abcdef" for character in value.lower())
            ):
                raise ValueError(f"Direct training {name} binding is missing or invalid.")
    task_config = AssociativeRecallConfig(
        vocab_size=config.vocab_size,
        sliding_window=config.sliding_window,
        key_count=64,
        value_start=80,
        value_count=64,
    )
    model = DeepSeekV4ForCausalLM(config).to(device)
    probe_objective = CSAProbeObjective(
        head_dim=config.head_dim,
        query_dim=config.q_lora_rank,
        vocab_size=config.vocab_size,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    auxiliary_parameter_count = sum(parameter.numel() for parameter in probe_objective.parameters())
    trainable_parameters = [*model.parameters(), *probe_objective.parameters()]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    generator = torch.Generator().manual_seed(data_order_seed)
    history = []
    transcript_entries: list[dict[str, Any]] = []
    transcript_root = _initial_transcript_root(
        scale=args.scale,
        seed=args.seed,
        launch_nonce=args.launch_nonce,
        manifest_sha256=args.manifest_sha256,
        trainer_sha256=args.trainer_sha256,
    )
    started = time.time()
    stopped_early = False
    _set_index_topk(model, config.index_topk)
    probe = CSASelectionProbe()

    for step in range(1, args.steps + 1):
        model.train()
        _set_index_topk(model, args.training_topk)
        length_index = int(
            torch.randint(0, len(TRAIN_SEQUENCE_LENGTHS), (1,), generator=generator).item()
        )
        sequence_length = TRAIN_SEQUENCE_LENGTHS[length_index]
        batch = generate_associative_recall_training_batch(
            task_config,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            num_queries=args.num_queries,
            generator=generator,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        probe.reset()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = _query_logits(
                model,
                batch.input_ids,
                batch.query_positions,
                csa_probe=probe,
            )
            answer_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                batch.targets.reshape(-1),
            )
            probe_loss = probe_objective(
                probe.records,
                query_positions=batch.query_positions,
                evidence_positions=batch.evidence_positions,
                targets=batch.targets,
            )
            loss = (
                answer_loss
                + args.ranking_loss_weight * probe_loss.ranking
                + args.read_ranking_loss_weight * probe_loss.read_ranking
                + args.value_loss_weight * probe_loss.value
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            args.gradient_clip_norm,
        )
        optimizer.step()
        transcript_entry = {
            "step": step,
            "optimizer_step": step,
            "sequence_length": sequence_length,
            "batch_sha256": _batch_digest(batch, sequence_length=sequence_length),
            "loss": float(loss.detach()),
            "answer_loss": float(answer_loss.detach()),
            "ranking_loss": float(probe_loss.ranking.detach()),
            "read_ranking_loss": float(probe_loss.read_ranking.detach()),
            "value_loss": float(probe_loss.value.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }
        if not all(
            math.isfinite(cast(float, value))
            for name, value in transcript_entry.items()
            if name
            in {
                "loss",
                "answer_loss",
                "ranking_loss",
                "read_ranking_loss",
                "value_loss",
                "gradient_norm",
            }
        ):
            raise ValueError("Direct training produced a non-finite transcript value.")
        transcript_root = _extend_transcript_root(transcript_root, transcript_entry)
        transcript_entries.append(transcript_entry)

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            full_memory = evaluate(
                model,
                task_config,
                sequence_lengths=SEQUENCE_LENGTHS,
                batches_per_length=args.eval_batches,
                batch_size=args.eval_batch_size,
                seed=training_evaluation_seed,
                topk=args.training_topk,
                device=device,
            )
            native = evaluate(
                model,
                task_config,
                sequence_lengths=SEQUENCE_LENGTHS,
                batches_per_length=args.eval_batches,
                batch_size=args.eval_batch_size,
                seed=training_evaluation_seed,
                topk=config.index_topk,
                device=device,
            )
            local = evaluate(
                model,
                task_config,
                sequence_lengths=SEQUENCE_LENGTHS,
                batches_per_length=args.eval_batches,
                batch_size=args.eval_batch_size,
                seed=training_evaluation_seed,
                topk=0,
                device=device,
            )
            _set_index_topk(model, config.index_topk)
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "answer_loss": float(answer_loss.detach()),
                "ranking_loss": float(probe_loss.ranking.detach()),
                "read_ranking_loss": float(probe_loss.read_ranking.detach()),
                "value_loss": float(probe_loss.value.detach()),
                "ranking_accuracy": float(probe_loss.ranking_accuracy.detach()),
                "read_ranking_accuracy": float(probe_loss.read_ranking_accuracy.detach()),
                "value_accuracy": float(probe_loss.value_accuracy.detach()),
                "probe_layers": probe_loss.layers,
                "gradient_norm": float(gradient_norm.detach()),
                "full_memory": full_memory,
                "native": native,
                "local_only": local,
                "native_minus_local": native["mean_accuracy"] - local["mean_accuracy"],
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if (
                not args.disable_early_stop
                and step >= args.minimum_steps
                and native["mean_accuracy"] >= args.target_accuracy
                and native["mean_accuracy"] - local["mean_accuracy"] >= args.minimum_memory_gap
            ):
                stopped_early = True
                break

    final_step = history[-1]["step"]
    checkpoint = None
    if not args.no_save:
        checkpoint_path = args.output_dir / f"{args.scale}-step-{final_step}.pt"
        checkpoint_provenance = {
            "schema_version": 1,
            "experiment_id": args.experiment_id,
            "scale": args.scale,
            "training_seed": args.seed,
            "initialization_seed": initialization_seed,
            "data_order_seed": data_order_seed,
            "training_evaluation_seed": training_evaluation_seed,
            "steps_completed": final_step,
            "minimum_steps": args.minimum_steps,
            "launch_nonce": args.launch_nonce,
            "canonical_trainer_sha256": args.trainer_sha256,
            "attestation_key_id": args.attestation_key_id,
            "training_hyperparameters": _args_direct_hyperparameters(args),
            "transcript_scheme": DIRECT_TRANSCRIPT_SCHEME,
            "transcript_entry_count": len(transcript_entries),
            "transcript_root": transcript_root,
            "source": source,
            "manifest_sha256": args.manifest_sha256,
            "implementation_digest": args.implementation_digest,
            "implementation_source_commit": args.implementation_source_commit,
            "execution_environment": stable_execution_environment,
        }
        checkpoint = _save_checkpoint_atomically(
            model,
            probe_objective,
            optimizer,
            config,
            checkpoint_path,
            provenance=checkpoint_provenance,
        )
    torch.cuda.synchronize()
    result = {
        "schema_version": (
            DIRECT_SUMMARY_SCHEMA_VERSION
            if args.experiment_id == DIRECT_TRAINING_EXPERIMENT_ID
            else 1
        ),
        "experiment_id": args.experiment_id,
        "scale": args.scale,
        "seed": args.seed,
        "initialization_seed": initialization_seed,
        "data_order_seed": data_order_seed,
        "training_evaluation_seed": training_evaluation_seed,
        "source": source,
        "command": [sys.executable, *sys.argv],
        "parameters": parameter_count,
        "auxiliary_parameters": auxiliary_parameter_count,
        "config": asdict(config),
        "task_config": asdict(task_config),
        "sequence_lengths": SEQUENCE_LENGTHS,
        "training_sequence_lengths": TRAIN_SEQUENCE_LENGTHS,
        "num_queries_per_training_sequence": args.num_queries,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "ranking_loss_weight": args.ranking_loss_weight,
        "read_ranking_loss_weight": args.read_ranking_loss_weight,
        "value_loss_weight": args.value_loss_weight,
        "training_topk": args.training_topk,
        "steps_completed": final_step,
        "stopped_early": stopped_early,
        "history": history,
        "training_hyperparameters": _args_direct_hyperparameters(args),
        "training_transcript": {
            "scheme": DIRECT_TRANSCRIPT_SCHEME,
            "entry_count": len(transcript_entries),
            "entries": transcript_entries,
            "root": transcript_root,
        },
        "checkpoint": checkpoint,
        "execution_environment": stable_execution_environment,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": execution_environment.selected_device_class(stable_execution_environment)[
                "name"
            ],
            **execution_environment.selected_device_context(stable_execution_environment),
            "elapsed_seconds": time.time() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    if args.experiment_id == DIRECT_TRAINING_EXPERIMENT_ID:
        if _source_state() != source:
            raise ValueError("Direct training source changed before final publication.")
        result["direct_training_contract"] = _direct_training_contract(args, source=source)
        result["payload_sha256"] = _canonical_json_digest(result)
        if trust_root is None:
            raise RuntimeError("Direct training lost its attestation trust root.")
        result["attestation"] = attestation.attest_payload(
            result,
            trust_root=trust_root,
            purpose=DIRECT_SUMMARY_ATTESTATION_PURPOSE,
        )
    summary_path = args.output_dir / f"{args.scale}-training.summary.json"
    _atomic_write_json(summary_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", default="m1-tier-s-associative-recall-v1")
    parser.add_argument("--scale", choices=tuple(SCALE_OVERRIDES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--minimum-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-queries", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--read-ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--training-topk", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--target-accuracy", type=float, default=0.85)
    parser.add_argument("--minimum-memory-gap", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-device-routing-identity-json", default="")
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--manifest-experiment-id", default="")
    parser.add_argument("--manifest-sha256", default="")
    parser.add_argument("--implementation-digest", default="")
    parser.add_argument("--implementation-source-commit", default="")
    parser.add_argument("--attestation-key-id", default="")
    parser.add_argument("--launch-nonce", default="")
    parser.add_argument("--trainer-sha256", default="")
    parser.add_argument("--direct-hyperparameters-json", default="")
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    result = train(args)
    print(
        json.dumps(
            {
                "scale": result["scale"],
                "steps_completed": result["steps_completed"],
                "checkpoint": result["checkpoint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
