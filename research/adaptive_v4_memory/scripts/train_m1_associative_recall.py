from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

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


def _save_checkpoint_atomically(
    model: DeepSeekV4ForCausalLM,
    config: DeepSeekV4Config,
    path: Path,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=".tier-s-",
            suffix=".pt.tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save({"config": asdict(config), "model": state}, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def train(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Tier-S training requires a CUDA device.")
    initialization_seed = args.seed
    data_order_seed = args.seed + 1
    training_evaluation_seed = args.seed + 10_000
    torch.manual_seed(initialization_seed)
    device = torch.device("cuda")
    config = build_config(args.scale)
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
                step >= args.minimum_steps
                and native["mean_accuracy"] >= args.target_accuracy
                and native["mean_accuracy"] - local["mean_accuracy"] >= args.minimum_memory_gap
            ):
                stopped_early = True
                break

    final_step = history[-1]["step"]
    checkpoint = None
    if not args.no_save:
        checkpoint_path = args.output_dir / f"{args.scale}-step-{final_step}.pt"
        checkpoint_sha256 = _save_checkpoint_atomically(model, config, checkpoint_path)
        checkpoint = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_path.stat().st_size,
        }
    torch.cuda.synchronize()
    result = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "scale": args.scale,
        "seed": args.seed,
        "initialization_seed": initialization_seed,
        "data_order_seed": data_order_seed,
        "training_evaluation_seed": training_evaluation_seed,
        "source": _source_state(),
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
        "checkpoint": checkpoint,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.time() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.scale}-training.summary.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-id", default="m1-tier-s-associative-recall-v1"
    )
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
