from __future__ import annotations

import copy
import json
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "research/adaptive_v4_memory/scripts"
sys.path.insert(0, str(SCRIPTS))

import p2_direct_attestation as attestation  # noqa: E402
import run_p2_direct_training_matrix as matrix  # noqa: E402


def _execution_environment() -> dict[str, Any]:
    return {
        "schema_version": matrix.execution_environment.SCHEMA_VERSION,
        "python_implementation": "CPython",
        "python_version": "3.12.0",
        "python_executable": str(Path(sys.executable).resolve()),
        "torch_version": "2.7.0+cu128",
        "cuda_runtime_version": "12.8",
        "cuda_driver_version": "570.00",
        "cuda_visible_devices": "0",
        "platform_system": "Linux",
        "platform_release": "test-kernel",
        "platform_machine": "x86_64",
        "platform_string": "Linux-test-x86_64",
        "current_device_index": 0,
        "visible_device_count": 1,
        "visible_devices": [
            {
                "logical_index": 0,
                "name": "Test CUDA GPU",
                "uuid": "GPU-test-0",
                "pci_bus_id": "0000:01:00.0",
                "compute_capability": [9, 0],
                "total_memory_bytes": 80 * 1024**3,
            }
        ],
    }


@dataclass(frozen=True)
class FrozenEnvironment:
    manifest_path: Path
    train_script: Path
    key_path: Path
    trust_root: attestation.TrustRoot
    context: matrix.FrozenContext
    trainer: matrix.CanonicalTrainerSnapshot
    execution_environment: dict[str, Any]


@dataclass(frozen=True)
class PreheldoutAdmissionCase:
    output_root: Path
    current_matrix_summary: Path
    superseded_manifest: Path
    superseded_matrix: Path
    preserved_claim: Path
    training_summary: Path
    checkpoint: Path


@dataclass
class FakeGPULease:
    path: Path
    events: list[str]
    file_descriptor: int = 2
    device: int = 1
    inode: int = 2
    closed: bool = False

    def assert_held(self) -> None:
        assert not self.closed
        self.events.append("assert-held")

    def fileno(self) -> int:
        self.assert_held()
        return self.file_descriptor

    def close(self) -> None:
        assert not self.closed
        self.events.append("close")
        self.closed = True


@dataclass
class FakeGPUController:
    events: list[str]
    leases: list[FakeGPULease]


def _load_or_create_preheldout_for_test(
    *,
    output_root: Path,
    matrix_summary: Path,
    context: matrix.FrozenContext,
    trust_root: attestation.TrustRoot,
    trainer_binding: dict[str, Any],
    expected_execution_environment: dict[str, Any],
    trainer_subprocesses_started: int,
) -> matrix.ValidatedPreheldoutAdmission | None:
    scheduler_lease = FakeGPULease(path=output_root.parent / "scheduler.lock", events=[])
    device_guard = FakeGPULease(path=output_root.parent / "device.lock", events=[])
    with matrix._matrix_lock(output_root) as matrix_lock:
        return matrix._load_or_create_preheldout_admission(
            output_root=output_root,
            matrix_summary=matrix_summary,
            context=context,
            trust_root=trust_root,
            trainer_binding=trainer_binding,
            expected_execution_environment=expected_execution_environment,
            trainer_subprocesses_started=trainer_subprocesses_started,
            matrix_lock=matrix_lock,
            gpu_lease=scheduler_lease,
            device_guard=device_guard,
        )


@pytest.fixture(autouse=True)
def fake_gpu_lease(monkeypatch: pytest.MonkeyPatch) -> FakeGPUController:
    monkeypatch.setattr(matrix, "REQUIRE_PREHELDOUT_ADMISSION", False)
    controller = FakeGPUController(events=[], leases=[])

    def acquire(label: str, *, path: Path) -> FakeGPULease:
        controller.events.append(f"acquire:{label}:{path}")
        lease = FakeGPULease(
            path=Path(path),
            events=controller.events,
            file_descriptor=2,
        )
        controller.leases.append(lease)
        return lease

    def acquire_device_guard(
        label: str,
        routing_identity: dict[str, Any],
    ) -> FakeGPULease:
        path = matrix.gpu_lock.canonical_device_guard_path(routing_identity)
        controller.events.append(f"acquire-device:{label}:{path}")
        lease = FakeGPULease(path=path, events=controller.events, file_descriptor=3)
        controller.leases.append(lease)
        return lease

    monkeypatch.setattr(matrix.gpu_lock, "acquire_gpu_lock", acquire)
    monkeypatch.setattr(matrix.gpu_lock, "acquire_device_guard", acquire_device_guard)
    return controller


def _tiny_config(_scale: str) -> matrix.trainer.DeepSeekV4Config:
    return matrix.trainer.DeepSeekV4Config(
        vocab_size=128,
        hidden_size=32,
        moe_intermediate_size=48,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        num_hash_layers=0,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        sliding_window=4,
        o_groups=2,
        o_lora_rank=8,
        index_n_heads=4,
        index_head_dim=4,
        index_topk=4,
        num_nextn_predict_layers=0,
    )


@pytest.fixture
def frozen_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FrozenEnvironment:
    monkeypatch.setattr(matrix.trainer, "build_config", _tiny_config)
    stable_environment = _execution_environment()
    monkeypatch.setattr(
        matrix.execution_environment,
        "capture_execution_environment",
        lambda: copy.deepcopy(stable_environment),
    )
    key_path = tmp_path / "paper-grade-attestation.key"
    key_path.write_bytes(bytes(range(32)))
    key_path.chmod(0o600)
    trust_root = attestation.load_trust_root(
        key_path,
        repository_root=matrix.REPOSITORY_ROOT,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"frozen":true}\n', encoding="utf-8")
    train_script = tmp_path / "train_m1_associative_recall.py"
    train_script.write_text("# canonical test trainer\n", encoding="utf-8")
    source = {"commit": "b" * 40, "dirty": False}
    implementation_digest = "a" * 64
    manifest = {
        "schema_version": 1,
        "experiment_id": matrix.contract.EXPERIMENT_ID,
        "implementation": {
            "tree_digest": implementation_digest,
            "source_commit": "c" * 40,
        },
        "attestation": attestation.public_manifest_contract(trust_root.key_id),
    }

    def validate_manifest(
        payload: dict[str, Any], *, verify_implementation: bool = True
    ) -> dict[str, Any]:
        assert payload == {"frozen": True}
        assert verify_implementation is True
        return manifest

    monkeypatch.setattr(matrix.contract, "validate_manifest_payload", validate_manifest)
    monkeypatch.setattr(matrix.contract, "source_state", lambda: dict(source))
    monkeypatch.setattr(
        matrix.contract,
        "implementation_tree_digest",
        lambda: implementation_digest,
    )
    monkeypatch.setattr(matrix, "TRAIN_SCRIPT", train_script)
    context = matrix.establish_frozen_context(manifest_path)
    opened, trainer_snapshot = matrix._open_canonical_trainer(train_script)
    opened.close()
    return FrozenEnvironment(
        manifest_path=manifest_path,
        train_script=train_script,
        key_path=key_path,
        trust_root=trust_root,
        context=context,
        trainer=trainer_snapshot,
        execution_environment=stable_environment,
    )


def _command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def _optimizer_state(
    model: matrix.trainer.DeepSeekV4ForCausalLM,
    probe: matrix.trainer.CSAProbeObjective,
) -> dict[str, Any]:
    model_named_parameters = list(model.named_parameters())
    probe_named_parameters = list(probe.named_parameters())
    parameters = [
        *(parameter for _name, parameter in model_named_parameters),
        *(parameter for _name, parameter in probe_named_parameters),
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=matrix.FROZEN_HYPERPARAMETERS["learning_rate"],
        weight_decay=matrix.FROZEN_HYPERPARAMETERS["weight_decay"],
        fused=True,
    )
    active_parameters = [
        parameter
        for name, parameter in model_named_parameters
        if not name.startswith("mtp_modules.") and ".self_attn.hca." not in name
    ]
    active_parameters.extend(parameter for _name, parameter in probe_named_parameters)
    for parameter in active_parameters:
        optimizer.state[parameter] = {
            "step": torch.tensor(float(matrix.STEPS)),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    return optimizer.state_dict()


def _optimizer_validation_case() -> tuple[
    dict[str, Any],
    int,
    set[int],
    tuple[tuple[int, int], ...],
    tuple[tuple[str, torch.nn.Parameter], ...],
]:
    config = _tiny_config("s55")
    model = matrix.trainer.DeepSeekV4ForCausalLM(config)
    probe = matrix.trainer.CSAProbeObjective(
        head_dim=config.head_dim,
        query_dim=config.q_lora_rank,
        vocab_size=config.vocab_size,
    )
    model_named_parameters = tuple(model.named_parameters())
    probe_named_parameters = tuple(probe.named_parameters())
    optimizer_named_parameters = (
        *model_named_parameters,
        *((f"probe_objective.{name}", parameter) for name, parameter in probe_named_parameters),
    )
    parameter_count = len(optimizer_named_parameters)
    expected_state_parameter_ids = {
        index
        for index, (name, _parameter) in enumerate(model_named_parameters)
        if not name.startswith("mtp_modules.") and ".self_attn.hca." not in name
    }
    expected_state_parameter_ids.update(range(len(model_named_parameters), parameter_count))
    routed_pairs = matrix._routed_expert_parameter_pairs(
        model_named_parameters,
        expected_state_parameter_ids=expected_state_parameter_ids,
    )
    return (
        _optimizer_state(model, probe),
        parameter_count,
        expected_state_parameter_ids,
        routed_pairs,
        optimizer_named_parameters,
    )


def _validate_optimizer_case(
    optimizer: dict[str, Any],
    *,
    parameter_count: int,
    expected_state_parameter_ids: set[int],
    routed_pairs: tuple[tuple[int, int], ...],
    expected_named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
) -> None:
    matrix._validate_optimizer_state(
        optimizer,
        expected_parameter_count=parameter_count,
        expected_state_parameter_ids=expected_state_parameter_ids,
        expected_routed_expert_parameter_pairs=routed_pairs,
        expected_named_parameters=expected_named_parameters,
    )


def _training_transcript(
    env: FrozenEnvironment,
    *,
    scale: str,
    seed: int,
    launch_nonce: str,
) -> dict[str, Any]:
    root = matrix.trainer._initial_transcript_root(
        scale=scale,
        seed=seed,
        launch_nonce=launch_nonce,
        manifest_sha256=str(env.context.manifest_binding["sha256"]),
        trainer_sha256=env.trainer.sha256,
    )
    entries: list[dict[str, Any]] = []
    for step in range(1, matrix.STEPS + 1):
        entry = {
            "step": step,
            "optimizer_step": step,
            "sequence_length": 64,
            "batch_sha256": matrix.contract.json_digest({"step": step, "seed": seed}),
            "loss": 1.0,
            "answer_loss": 0.5,
            "ranking_loss": 0.2,
            "read_ranking_loss": 0.2,
            "value_loss": 0.1,
            "gradient_norm": 0.75,
        }
        root = matrix.trainer._extend_transcript_root(root, entry)
        entries.append(entry)
    return {
        "scheme": matrix.trainer.DIRECT_TRANSCRIPT_SCHEME,
        "entry_count": matrix.STEPS,
        "entries": entries,
        "root": root,
    }


def _history() -> list[dict[str, Any]]:
    metrics = {
        **{f"accuracy_length_{length}": 0.5 for length in matrix.trainer.SEQUENCE_LENGTHS},
        "mean_accuracy": 0.5,
    }
    return [
        {
            "step": step,
            "loss": 1.0,
            "answer_loss": 0.5,
            "ranking_loss": 0.2,
            "read_ranking_loss": 0.2,
            "value_loss": 0.1,
            "ranking_accuracy": 0.5,
            "read_ranking_accuracy": 0.5,
            "value_accuracy": 0.5,
            "probe_layers": 2,
            "gradient_norm": 0.75,
            "full_memory": dict(metrics),
            "native": dict(metrics),
            "local_only": dict(metrics),
            "native_minus_local": 0.0,
            "elapsed_seconds": float(index),
        }
        for index, step in enumerate(matrix.EVALUATION_STEPS)
    ]


def _write_valid_summary(
    *,
    output_root: Path,
    env: FrozenEnvironment,
    scale: str,
    seed: int,
    launch_nonce: str,
    command: list[str] | None = None,
    authenticated: bool = True,
) -> tuple[Path, dict[str, Any]]:
    output_dir = output_root / scale / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript = _training_transcript(
        env,
        scale=scale,
        seed=seed,
        launch_nonce=launch_nonce,
    )
    config = matrix.trainer.build_config(scale)
    model = matrix.trainer.DeepSeekV4ForCausalLM(config)
    probe = matrix.trainer.CSAProbeObjective(
        head_dim=config.head_dim,
        query_dim=config.q_lora_rank,
        vocab_size=config.vocab_size,
    )
    checkpoint_path = output_dir / f"{scale}-step-{matrix.STEPS}.pt"
    torch.save(
        {
            "checkpoint_schema_version": matrix.trainer.DIRECT_CHECKPOINT_SCHEMA_VERSION,
            "provenance": matrix._checkpoint_provenance(
                env.context,
                scale=scale,
                seed=seed,
                launch_nonce=launch_nonce,
                trainer_sha256=env.trainer.sha256,
                transcript_root=transcript["root"],
                execution_environment_binding=env.execution_environment,
            ),
            "config": matrix.asdict(config),
            "model": model.state_dict(),
            "probe_objective": probe.state_dict(),
            "optimizer": _optimizer_state(model, probe),
        },
        checkpoint_path,
    )
    if command is None:
        command = matrix.build_training_command(
            train_script=env.train_script,
            output_root=output_root,
            scale=scale,
            seed=seed,
            context=env.context,
            launch_nonce=launch_nonce,
            trainer_sha256=env.trainer.sha256,
            device_routing_identity=(
                matrix.execution_environment.selected_device_routing_identity(
                    env.execution_environment
                )
            ),
        )
    task_config = {
        "vocab_size": config.vocab_size,
        "num_pairs": 10,
        "key_start": 16,
        "key_count": 64,
        "value_start": 80,
        "value_count": 64,
        "distractor_start": 1024,
        "separator_token_id": 3,
        "query_token_id": 4,
        "bos_token_id": 2,
        "sliding_window": config.sliding_window,
    }
    payload: dict[str, Any] = {
        "schema_version": matrix.trainer.DIRECT_SUMMARY_SCHEMA_VERSION,
        "experiment_id": matrix.EXPERIMENT_ID,
        "scale": scale,
        "seed": seed,
        **matrix._seed_values(seed),
        "source": env.context.source,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "auxiliary_parameters": sum(parameter.numel() for parameter in probe.parameters()),
        "config": matrix.asdict(config),
        "task_config": task_config,
        "sequence_lengths": list(matrix.trainer.SEQUENCE_LENGTHS),
        "training_sequence_lengths": list(matrix.trainer.TRAIN_SEQUENCE_LENGTHS),
        "num_queries_per_training_sequence": matrix.FROZEN_HYPERPARAMETERS["num_queries"],
        "batch_size": matrix.FROZEN_HYPERPARAMETERS["batch_size"],
        "learning_rate": matrix.FROZEN_HYPERPARAMETERS["learning_rate"],
        "weight_decay": matrix.FROZEN_HYPERPARAMETERS["weight_decay"],
        "ranking_loss_weight": matrix.FROZEN_HYPERPARAMETERS["ranking_loss_weight"],
        "read_ranking_loss_weight": matrix.FROZEN_HYPERPARAMETERS["read_ranking_loss_weight"],
        "value_loss_weight": matrix.FROZEN_HYPERPARAMETERS["value_loss_weight"],
        "training_topk": matrix.FROZEN_HYPERPARAMETERS["training_topk"],
        "training_hyperparameters": matrix.FROZEN_HYPERPARAMETERS,
        "command": command,
        "steps_completed": matrix.STEPS,
        "stopped_early": False,
        "history": _history(),
        "training_transcript": transcript,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": matrix._sha256(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
        },
        "execution_environment": copy.deepcopy(env.execution_environment),
        "runtime": {
            "python": env.execution_environment["python_version"],
            "torch": env.execution_environment["torch_version"],
            "device": env.execution_environment["visible_devices"][0]["name"],
            **matrix.execution_environment.selected_device_context(env.execution_environment),
            "elapsed_seconds": 123.0,
            "peak_allocated_bytes": 1024,
            "peak_reserved_bytes": 2048,
        },
    }
    if authenticated:
        payload["direct_training_contract"] = matrix._direct_training_binding(
            env.context,
            launch_nonce=launch_nonce,
            trainer_sha256=env.trainer.sha256,
        )
        payload = matrix._attested_payload(
            payload,
            trust_root=env.trust_root,
            purpose=matrix.SUMMARY_ATTESTATION_PURPOSE,
        )
    summary_path = output_dir / f"{scale}-training.summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary_path, payload


def _prepare_preheldout_admission_case(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env: FrozenEnvironment,
) -> PreheldoutAdmissionCase:
    monkeypatch.setattr(matrix, "REQUIRE_PREHELDOUT_ADMISSION", True)
    output_root = tmp_path / "paper_grade" / "p2_post_rank_direct" / "training"
    output_root.mkdir(parents=True)
    lock_path = output_root.parent / matrix.LOCK_NAME
    lock_path.write_bytes(b"")
    lock_path.chmod(0o600)
    implementation_commit = "d" * 40
    implementation_digest = "e" * 64
    attempt_commit = "f" * 40
    superseded_manifest = tmp_path / "p2-direct-controller-manifest-v1.json"
    manifest_payload = dict.fromkeys(matrix.contract.MANIFEST_TOP_LEVEL_FIELDS)
    manifest_payload.update(
        {
            "schema_version": 1,
            "experiment_id": "p2-post-rank-direct-controller-v1",
            "status": "frozen_before_any_fresh_direct_controller_quality_result",
            "attestation": attestation.public_manifest_contract(env.trust_root.key_id),
            "implementation": {
                "paths": [
                    matrix.contract.PROJECT_DEPENDENCY_SPEC_PATH,
                    matrix.contract.PACKAGE_IMPLEMENTATION_ROOT,
                    *matrix.contract.DIRECT_RESEARCH_IMPLEMENTATION_PATHS,
                ],
                "tree_digest": implementation_digest,
                "source_commit": implementation_commit,
            },
        }
    )
    superseded_manifest.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(matrix.contract, "SUPERSEDED_MANIFEST_PATH", str(superseded_manifest))
    monkeypatch.setattr(
        matrix.contract,
        "SUPERSEDED_MANIFEST_SHA256",
        matrix._sha256(superseded_manifest),
    )
    monkeypatch.setattr(
        matrix.contract, "SUPERSEDED_IMPLEMENTATION_SOURCE_COMMIT", implementation_commit
    )
    monkeypatch.setattr(
        matrix.contract, "SUPERSEDED_IMPLEMENTATION_TREE_DIGEST", implementation_digest
    )
    monkeypatch.setattr(matrix.contract, "SUPERSEDED_ATTEMPT_SOURCE_COMMIT", attempt_commit)
    monkeypatch.setattr(
        matrix.contract,
        "superseded_v1_implementation_tree_digest_at_commit",
        lambda: implementation_digest,
    )
    monkeypatch.setattr(
        matrix,
        "_superseded_attempt_is_ancestor_of_head",
        lambda commit: commit == attempt_commit,
    )

    manifest_binding = matrix._manifest_binding(superseded_manifest, manifest_payload)
    old_context = matrix.FrozenContext(
        manifest_path=superseded_manifest.resolve(),
        manifest_binding=manifest_binding,
        source={"commit": attempt_commit, "dirty": False},
    )
    old_env = FrozenEnvironment(
        manifest_path=superseded_manifest,
        train_script=env.train_script,
        key_path=env.key_path,
        trust_root=env.trust_root,
        context=old_context,
        trainer=env.trainer,
        execution_environment=copy.deepcopy(env.execution_environment),
    )
    scale, seed = matrix.PREHELDOUT_ADMISSION_COORDINATE
    launch_nonce = "1" * 64
    training_summary, summary_payload = _write_valid_summary(
        output_root=output_root,
        env=old_env,
        scale=scale,
        seed=seed,
        launch_nonce=launch_nonce,
    )
    checkpoint = Path(summary_payload["checkpoint"]["path"])
    preserved_claim = training_summary.parent / matrix.CLAIM_NAME
    preserved_claim.write_text(f"{launch_nonce}\n", encoding="ascii")

    routing_identity = matrix.execution_environment.selected_device_routing_identity(
        env.execution_environment
    )
    device_guard = FakeGPULease(
        path=matrix.gpu_lock.canonical_device_guard_path(routing_identity),
        events=[],
        file_descriptor=3,
    )
    gpu_lease_binding = matrix._gpu_lease_binding(
        tmp_path / "training-gpu.lock",
        frozen_execution_environment=env.execution_environment,
        device_guard=device_guard,
    )
    old_matrix_semantic = {
        "schema_version": 3,
        "experiment_id": matrix.EXPERIMENT_ID,
        "artifact_type": matrix.ARTIFACT_TYPE,
        "status": "in_progress",
        "source": old_context.source,
        "manifest": manifest_binding,
        "attestation_contract": manifest_binding["attestation"],
        "canonical_trainer": env.trainer.public_binding,
        "gpu_lease": gpu_lease_binding,
        "execution_environment": copy.deepcopy(env.execution_environment),
        "scales": list(matrix.FROZEN_SCALES),
        "frozen_training_seeds": list(matrix.FROZEN_TRAINING_SEEDS),
        "steps": matrix.STEPS,
        "minimum_steps": matrix.MINIMUM_STEPS,
        "seed_rules": matrix.SEED_RULES,
        "expected_runs": len(matrix.FROZEN_SCALES) * len(matrix.FROZEN_TRAINING_SEEDS),
        "completed_runs": 0,
        "runs": [],
    }
    old_matrix_payload = matrix._attested_payload(
        old_matrix_semantic,
        trust_root=env.trust_root,
        purpose=matrix.SUPERSEDED_MATRIX_ATTESTATION_PURPOSE,
    )
    superseded_matrix = output_root / matrix.SUPERSEDED_MATRIX_SUMMARY_NAME
    superseded_matrix.write_text(
        json.dumps(old_matrix_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        matrix.contract, "SUPERSEDED_MATRIX_SHA256", matrix._sha256(superseded_matrix)
    )
    monkeypatch.setattr(matrix.contract, "SUPERSEDED_CLAIM_SHA256", matrix._sha256(preserved_claim))
    monkeypatch.setattr(
        matrix.contract,
        "SUPERSEDED_TRAINING_SUMMARY_SHA256",
        matrix._sha256(training_summary),
    )
    monkeypatch.setattr(matrix.contract, "SUPERSEDED_CHECKPOINT_SHA256", matrix._sha256(checkpoint))
    return PreheldoutAdmissionCase(
        output_root=output_root,
        current_matrix_summary=output_root / matrix.MATRIX_SUMMARY.name,
        superseded_manifest=superseded_manifest,
        superseded_matrix=superseded_matrix,
        preserved_claim=preserved_claim,
        training_summary=training_summary,
        checkpoint=checkpoint,
    )


def _validate(
    payload: dict[str, Any],
    *,
    output_root: Path,
    env: FrozenEnvironment,
    launch_nonce: str,
    train_script: Path | None = None,
    trust_root: attestation.TrustRoot | None = None,
) -> Any:
    return matrix.validate_training_summary(
        payload,
        output_root=output_root,
        train_script=env.train_script if train_script is None else train_script,
        context=env.context,
        scale="s55",
        seed=6071406,
        trust_root=env.trust_root if trust_root is None else trust_root,
        launch_nonce=launch_nonce,
        trainer_sha256=env.trainer.sha256,
    )


def test_routed_expert_optimizer_update_counts_may_be_below_global_step() -> None:
    optimizer, parameter_count, expected_state_parameter_ids, routed_pairs, named_parameters = (
        _optimizer_validation_case()
    )
    assert routed_pairs
    gate_id, down_id = routed_pairs[0]
    optimizer["state"][gate_id]["step"] = torch.tensor(728.0)
    optimizer["state"][down_id]["step"] = torch.tensor(728.0)

    _validate_optimizer_case(
        optimizer,
        parameter_count=parameter_count,
        expected_state_parameter_ids=expected_state_parameter_ids,
        routed_pairs=routed_pairs,
        expected_named_parameters=named_parameters,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("lower-dense", "Non-routed optimizer parameter"),
        ("zero-routed", "outside the frozen step range"),
        ("above-routed", "outside the frozen step range"),
        ("fractional-routed", "finite integer"),
        ("nonfinite-routed", "non-finite tensor"),
        ("mismatched-pair", "parameter-pair update counts differ"),
        ("partial-pair", "state inventory drifted"),
        ("missing-state-field", "parameter-state schema drifted"),
        ("extra-state-field", "parameter-state schema drifted"),
        ("moment-shape", "exp_avg tensor contract drifted"),
        ("moment-dtype", "exp_avg_sq tensor contract drifted"),
    ],
)
def test_optimizer_update_count_adversarial_cases_fail_closed(
    mutation: str,
    message: str,
) -> None:
    optimizer, parameter_count, expected_state_parameter_ids, routed_pairs, named_parameters = (
        _optimizer_validation_case()
    )
    routed_ids = {parameter_id for pair in routed_pairs for parameter_id in pair}
    gate_id, down_id = routed_pairs[0]
    if mutation == "lower-dense":
        dense_id = next(iter(expected_state_parameter_ids - routed_ids))
        optimizer["state"][dense_id]["step"] = torch.tensor(999.0)
    elif mutation == "zero-routed":
        optimizer["state"][gate_id]["step"] = torch.tensor(0.0)
        optimizer["state"][down_id]["step"] = torch.tensor(0.0)
    elif mutation == "above-routed":
        optimizer["state"][gate_id]["step"] = torch.tensor(float(matrix.STEPS + 1))
        optimizer["state"][down_id]["step"] = torch.tensor(float(matrix.STEPS + 1))
    elif mutation == "fractional-routed":
        optimizer["state"][gate_id]["step"] = torch.tensor(728.5)
        optimizer["state"][down_id]["step"] = torch.tensor(728.5)
    elif mutation == "nonfinite-routed":
        optimizer["state"][gate_id]["step"] = torch.tensor(float("inf"))
        optimizer["state"][down_id]["step"] = torch.tensor(float("inf"))
    elif mutation == "mismatched-pair":
        optimizer["state"][gate_id]["step"] = torch.tensor(728.0)
        optimizer["state"][down_id]["step"] = torch.tensor(729.0)
    elif mutation == "partial-pair":
        del optimizer["state"][down_id]
    elif mutation == "missing-state-field":
        del optimizer["state"][gate_id]["exp_avg"]
    elif mutation == "extra-state-field":
        optimizer["state"][gate_id]["unregistered"] = torch.tensor(0.0)
    elif mutation == "moment-shape":
        parameter_id = next(
            index for index in expected_state_parameter_ids if named_parameters[index][1].ndim >= 2
        )
        parameter = named_parameters[parameter_id][1]
        optimizer["state"][parameter_id]["exp_avg"] = torch.zeros(
            parameter.numel(), dtype=parameter.dtype
        )
    else:
        optimizer["state"][gate_id]["exp_avg_sq"] = optimizer["state"][gate_id]["exp_avg_sq"].to(
            torch.float64
        )

    with pytest.raises(ValueError, match=message):
        _validate_optimizer_case(
            optimizer,
            parameter_count=parameter_count,
            expected_state_parameter_ids=expected_state_parameter_ids,
            routed_pairs=routed_pairs,
            expected_named_parameters=named_parameters,
        )


def test_preheldout_admission_is_one_shot_exact_byte_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    evidence_paths = (
        case.superseded_manifest,
        case.superseded_matrix,
        case.preserved_claim,
        case.training_summary,
        case.checkpoint,
    )
    evidence_before = {path: matrix._sha256(path) for path in evidence_paths}

    admission = _load_or_create_preheldout_for_test(
        output_root=case.output_root,
        matrix_summary=case.current_matrix_summary,
        context=env.context,
        trust_root=env.trust_root,
        trainer_binding=env.trainer.public_binding,
        expected_execution_environment=env.execution_environment,
        trainer_subprocesses_started=0,
    )

    assert admission is not None
    assert (admission.admitted_run_record["scale"], admission.admitted_run_record["seed"]) == (
        matrix.PREHELDOUT_ADMISSION_COORDINATE
    )
    assert admission.payload["preserved_claim"]["retention"] == (
        "registered-exact-byte-claim-must-remain-forever"
    )
    assert admission.payload["downstream_roots_absence_verified_immediately_before_creation"] == [
        str(path) for path in matrix._downstream_roots(case.output_root)
    ]
    assert admission.payload["scientific_child_processes_started_at_creation"] == 0
    assert admission.payload["read_only_git_provenance_commands_within_admission_creation"] == list(
        matrix.ADMISSION_READ_ONLY_GIT_PROVENANCE_COMMANDS
    )
    assert admission.payload[
        "direct_root_top_level_inventory_verified_immediately_before_creation"
    ] == list(matrix.ADMISSION_DIRECT_ROOT_TOP_LEVEL_INVENTORY)
    assert admission.payload["exclusive_matrix_lock_verified_immediately_before_creation"] is True
    assert admission.payload["scheduler_gpu_lease_verified_immediately_before_creation"] is True
    assert admission.payload["physical_device_guard_verified_immediately_before_creation"] is True
    admission_path = case.output_root / matrix.PREHELDOUT_ADMISSION_NAME
    admission_stat = admission_path.stat()
    admission_bytes = admission_path.read_bytes()
    replay = _load_or_create_preheldout_for_test(
        output_root=case.output_root,
        matrix_summary=case.current_matrix_summary,
        context=env.context,
        trust_root=env.trust_root,
        trainer_binding=env.trainer.public_binding,
        expected_execution_environment=env.execution_environment,
        trainer_subprocesses_started=0,
    )
    assert replay == admission
    assert admission_path.stat().st_ino == admission_stat.st_ino
    assert admission_path.read_bytes() == admission_bytes
    assert {path: matrix._sha256(path) for path in evidence_paths} == evidence_before

    admission_path.write_bytes(admission_bytes + b" ")
    with pytest.raises(ValueError, match="exact-byte encoding drifted"):
        matrix.load_preheldout_admission(
            output_root=case.output_root,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
        )


def test_admission_public_binding_rejects_path_replacement_before_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    admission = _load_or_create_preheldout_for_test(
        output_root=case.output_root,
        matrix_summary=case.current_matrix_summary,
        context=env.context,
        trust_root=env.trust_root,
        trainer_binding=env.trainer.public_binding,
        expected_execution_environment=env.execution_environment,
        trainer_subprocesses_started=0,
    )
    assert admission is not None
    original_binding = matrix._admission_public_binding

    def replace_then_bind(
        opened: attestation.OpenedRegularFile,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        replacement = opened.path.with_name("replacement-admission.json")
        replacement.write_text("{}\n", encoding="utf-8")
        replacement.chmod(0o600)
        matrix.os.replace(replacement, opened.path)
        return original_binding(opened, payload)

    monkeypatch.setattr(matrix, "_admission_public_binding", replace_then_bind)
    with pytest.raises(ValueError, match="changed during validation|replaced during validation"):
        matrix.load_preheldout_admission(
            output_root=case.output_root,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
        )


@pytest.mark.parametrize("downstream_index", range(6))
def test_every_direct_downstream_artifact_blocks_preheldout_admission(
    downstream_index: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    monkeypatch.setattr(matrix, "REQUIRE_PREHELDOUT_ADMISSION", True)
    output_root = tmp_path / "p2_post_rank_direct" / "training"
    output_root.mkdir(parents=True)
    downstream_roots = matrix._downstream_roots(output_root)
    assert tuple(path.name for path in downstream_roots) == (
        "calibration",
        "top_p_physical_match",
        "controller",
        "controller-integrity.json",
        "controller-summary.json",
        ".controller.p2-direct-controller-workers",
    )
    blocker = downstream_roots[downstream_index]
    if blocker.suffix == ".json":
        blocker.write_text("{}\n", encoding="utf-8")
    else:
        blocker.mkdir()

    with pytest.raises(ValueError, match="every direct downstream artifact"):
        _load_or_create_preheldout_for_test(
            output_root=output_root,
            matrix_summary=output_root / matrix.MATRIX_SUMMARY.name,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
            trainer_subprocesses_started=0,
        )


def test_arbitrary_direct_root_sibling_blocks_preheldout_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    rogue = case.output_root.parent / "rogue-quality.json"
    rogue.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unregistered direct artifact"):
        _load_or_create_preheldout_for_test(
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
            trainer_subprocesses_started=0,
        )


def test_admission_postflight_blocks_sibling_injected_during_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    original_write = matrix._exclusive_write_admission
    rogue = case.output_root.parent / "race-injected-quality.json"

    def inject_after_atomic_commit(path: Path, payload: dict[str, Any]) -> None:
        original_write(path, payload)
        rogue.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(matrix, "_exclusive_write_admission", inject_after_atomic_commit)
    with pytest.raises(ValueError, match="postflight"):
        _load_or_create_preheldout_for_test(
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
            trainer_subprocesses_started=0,
        )
    assert (case.output_root / matrix.PREHELDOUT_ADMISSION_NAME).is_file()
    assert not case.current_matrix_summary.exists()

    rogue.unlink()
    monkeypatch.setattr(matrix, "_exclusive_write_admission", original_write)
    recovered = _load_or_create_preheldout_for_test(
        output_root=case.output_root,
        matrix_summary=case.current_matrix_summary,
        context=env.context,
        trust_root=env.trust_root,
        trainer_binding=env.trainer.public_binding,
        expected_execution_environment=env.execution_environment,
        trainer_subprocesses_started=0,
    )
    assert recovered is not None


def test_atomic_admission_commit_recovers_exact_partial_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "direct" / "training" / matrix.PREHELDOUT_ADMISSION_NAME
    payload = {"schema_version": 1, "terminal": True}
    original_write = matrix.os.write
    writes = 0

    def partial_then_crash(descriptor: int, data: Any) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return original_write(descriptor, bytes(data[: max(1, len(data) // 2)]))
        raise OSError("injected write crash")

    monkeypatch.setattr(matrix.os, "write", partial_then_crash)
    with pytest.raises(OSError, match="injected write crash"):
        matrix._exclusive_write_admission(path, payload)
    assert not path.exists()
    staging = matrix._admission_staging_path(path)
    assert staging.is_file()

    monkeypatch.setattr(matrix.os, "write", original_write)
    matrix._exclusive_write_admission(path, payload)
    assert path.read_bytes() == matrix._admission_encoded_bytes(payload)
    assert not staging.exists()
    assert path.stat().st_nlink == 1


@pytest.mark.parametrize("fail_on_directory", [False, True])
def test_atomic_admission_commit_recovers_fsync_failure(
    fail_on_directory: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "direct" / "training" / matrix.PREHELDOUT_ADMISSION_NAME
    payload = {"schema_version": 1, "terminal": True}
    original_fsync = matrix.os.fsync
    failed = False

    def fail_once(descriptor: int) -> None:
        nonlocal failed
        is_directory = stat.S_ISDIR(matrix.os.fstat(descriptor).st_mode)
        if not failed and is_directory is fail_on_directory:
            failed = True
            raise OSError("injected fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(matrix.os, "fsync", fail_once)
    with pytest.raises(OSError, match="injected fsync failure"):
        matrix._exclusive_write_admission(path, payload)
    recovery_targets: list[str] = []

    def record_recovery_fsync(descriptor: int) -> None:
        mode = matrix.os.fstat(descriptor).st_mode
        recovery_targets.append("directory" if stat.S_ISDIR(mode) else "regular")
        original_fsync(descriptor)

    monkeypatch.setattr(matrix.os, "fsync", record_recovery_fsync)

    if path.exists():
        matrix._cleanup_committed_admission_staging(path, payload)
    else:
        matrix._exclusive_write_admission(path, payload)
    assert "regular" in recovery_targets
    assert path.read_bytes() == matrix._admission_encoded_bytes(payload)
    assert not matrix._admission_staging_path(path).exists()


def test_committed_admission_recovery_fsyncs_final_parent_before_staging_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "direct" / "training" / matrix.PREHELDOUT_ADMISSION_NAME
    payload = {"schema_version": 1, "terminal": True}
    original_fsync = matrix.os.fsync
    failed = False

    def fail_final_parent_once(descriptor: int) -> None:
        nonlocal failed
        mode = matrix.os.fstat(descriptor).st_mode
        descriptor_path = Path(matrix.os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
        if not failed and stat.S_ISDIR(mode) and descriptor_path == path.parent.resolve():
            failed = True
            raise OSError("injected final-parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(matrix.os, "fsync", fail_final_parent_once)
    with pytest.raises(OSError, match="final-parent fsync failure"):
        matrix._exclusive_write_admission(path, payload)
    staging = matrix._admission_staging_path(path)
    assert path.is_file() and staging.is_file()
    assert path.stat().st_ino == staging.stat().st_ino
    assert path.stat().st_nlink == 2

    recovery_directories: list[Path] = []

    def record_cleanup_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(matrix.os.fstat(descriptor).st_mode):
            recovery_directories.append(
                Path(matrix.os.readlink(f"/proc/self/fd/{descriptor}")).resolve()
            )
        original_fsync(descriptor)

    monkeypatch.setattr(matrix.os, "fsync", record_cleanup_fsync)
    matrix._cleanup_committed_admission_staging(path, payload)
    assert recovery_directories[:2] == [path.parent.resolve(), staging.parent.resolve()]
    assert path.stat().st_nlink == 1
    assert not staging.exists()


def test_preheldout_admission_precedes_every_trainer_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    monkeypatch.setattr(matrix, "REQUIRE_PREHELDOUT_ADMISSION", True)
    output_root = tmp_path / "training"
    output_root.mkdir()
    with pytest.raises(ValueError, match="precede every trainer subprocess"):
        _load_or_create_preheldout_for_test(
            output_root=output_root,
            matrix_summary=output_root / matrix.MATRIX_SUMMARY.name,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
            trainer_subprocesses_started=1,
        )


@pytest.mark.parametrize("lost_lease", ["matrix", "scheduler", "device"])
def test_preheldout_admission_requires_all_three_live_leases(
    lost_lease: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    scheduler = FakeGPULease(path=case.output_root.parent / "scheduler.lock", events=[])
    device = FakeGPULease(path=case.output_root.parent / "device.lock", events=[])
    if lost_lease == "scheduler":
        scheduler.closed = True
    if lost_lease == "device":
        device.closed = True

    if lost_lease == "matrix":
        with matrix._matrix_lock(case.output_root) as expired_matrix_lock:
            pass
        with pytest.raises((AssertionError, ValueError)):
            matrix._load_or_create_preheldout_admission(
                output_root=case.output_root,
                matrix_summary=case.current_matrix_summary,
                context=env.context,
                trust_root=env.trust_root,
                trainer_binding=env.trainer.public_binding,
                expected_execution_environment=env.execution_environment,
                trainer_subprocesses_started=0,
                matrix_lock=expired_matrix_lock,
                gpu_lease=scheduler,
                device_guard=device,
            )
        return

    with matrix._matrix_lock(case.output_root) as live_matrix_lock:
        with pytest.raises((AssertionError, ValueError)):
            matrix._load_or_create_preheldout_admission(
                output_root=case.output_root,
                matrix_summary=case.current_matrix_summary,
                context=env.context,
                trust_root=env.trust_root,
                trainer_binding=env.trainer.public_binding,
                expected_execution_environment=env.execution_environment,
                trainer_subprocesses_started=0,
                matrix_lock=live_matrix_lock,
                gpu_lease=scheduler,
                device_guard=device,
            )


@pytest.mark.parametrize(("returncode", "expected"), [(0, True), (1, False), (128, False)])
def test_superseded_attempt_ancestry_probe_is_exact(
    returncode: int,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    commit = "f" * 40
    assert matrix._superseded_attempt_is_ancestor_of_head(commit) is expected
    assert observed == {
        "command": ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        "cwd": matrix.REPOSITORY_ROOT,
        "check": False,
        "capture_output": True,
        "text": True,
    }


def test_preheldout_admission_rejects_ancestry_exact_bytes_and_old_hmac_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )

    monkeypatch.setattr(matrix, "_superseded_attempt_is_ancestor_of_head", lambda _commit: False)
    with pytest.raises(ValueError, match="not an ancestor of current HEAD"):
        _load_or_create_preheldout_for_test(
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
            trainer_subprocesses_started=0,
        )
    monkeypatch.setattr(matrix, "_superseded_attempt_is_ancestor_of_head", lambda _commit: True)

    for path in (
        case.superseded_manifest,
        case.superseded_matrix,
        case.preserved_claim,
        case.training_summary,
        case.checkpoint,
    ):
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="digest|bytes"):
            _load_or_create_preheldout_for_test(
                output_root=case.output_root,
                matrix_summary=case.current_matrix_summary,
                context=env.context,
                trust_root=env.trust_root,
                trainer_binding=env.trainer.public_binding,
                expected_execution_environment=env.execution_environment,
                trainer_subprocesses_started=0,
            )
        assert not (case.output_root / matrix.PREHELDOUT_ADMISSION_NAME).exists()
        path.write_bytes(original)

    original_matrix = case.superseded_matrix.read_bytes()
    original_matrix_sha256 = matrix.contract.SUPERSEDED_MATRIX_SHA256
    tampered = json.loads(original_matrix)
    tampered["completed_runs"] = 1
    digest_source = dict(tampered)
    digest_source.pop("attestation")
    digest_source.pop("payload_sha256")
    tampered["payload_sha256"] = matrix.contract.json_digest(digest_source)
    semantic = dict(tampered)
    envelope = semantic.pop("attestation")
    envelope["payload_sha256"] = attestation.checksum(semantic)
    case.superseded_matrix.write_text(
        json.dumps(tampered, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        matrix.contract,
        "SUPERSEDED_MATRIX_SHA256",
        matrix._sha256(case.superseded_matrix),
    )
    with pytest.raises(ValueError, match="MAC verification failed"):
        _load_or_create_preheldout_for_test(
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            expected_execution_environment=env.execution_environment,
            trainer_subprocesses_started=0,
        )
    case.superseded_matrix.write_bytes(original_matrix)
    monkeypatch.setattr(
        matrix.contract,
        "SUPERSEDED_MATRIX_SHA256",
        original_matrix_sha256,
    )


def test_grid_command_and_full_hyperparameters_are_frozen(
    tmp_path: Path, frozen_environment: FrozenEnvironment
) -> None:
    env = frozen_environment
    nonce = "1" * 64
    command = matrix.build_training_command(
        train_script=env.train_script,
        output_root=tmp_path / "training",
        scale="s151",
        seed=6071410,
        context=env.context,
        launch_nonce=nonce,
        trainer_sha256=env.trainer.sha256,
        device_routing_identity=matrix.execution_environment.selected_device_routing_identity(
            env.execution_environment
        ),
    )

    assert matrix.FROZEN_SCALES == ("s55", "s151")
    assert matrix.FROZEN_TRAINING_SEEDS == tuple(range(6071406, 6071411))
    assert matrix.STEPS == matrix.MINIMUM_STEPS == 1_000
    assert _command_value(command, "--steps") == "1000"
    assert _command_value(command, "--minimum-steps") == "1000"
    assert _command_value(command, "--device") == "cuda:0"
    assert json.loads(
        _command_value(command, "--expected-device-routing-identity-json")
    ) == matrix.execution_environment.selected_device_routing_identity(env.execution_environment)
    assert "--disable-early-stop" in command
    assert _command_value(command, "--launch-nonce") == nonce
    assert json.loads(_command_value(command, "--direct-hyperparameters-json")) == (
        matrix.FROZEN_HYPERPARAMETERS
    )


def test_training_command_validation_accepts_equivalent_output_root_spelling(
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    relative_root = Path("artifacts") / "command-spelling-regression" / "training"
    output_dir = matrix._run_output_dir(relative_root, "s55", 6071406)
    nonce = "1" * 64
    routing_identity = matrix.execution_environment.selected_device_routing_identity(
        env.execution_environment
    )
    command = matrix.build_training_command(
        train_script=env.train_script,
        output_root=relative_root,
        scale="s55",
        seed=6071406,
        context=env.context,
        launch_nonce=nonce,
        trainer_sha256=env.trainer.sha256,
        device_index=0,
        device_routing_identity=routing_identity,
    )

    matrix._validate_training_command(
        command,
        train_script=env.train_script,
        output_dir=output_dir.resolve(),
        scale="s55",
        seed=6071406,
        context=env.context,
        launch_nonce=nonce,
        trainer_sha256=env.trainer.sha256,
        device_index=0,
        device_routing_identity=routing_identity,
    )


def test_training_command_routes_to_nonzero_current_logical_device(
    tmp_path: Path, frozen_environment: FrozenEnvironment
) -> None:
    command = matrix.build_training_command(
        train_script=frozen_environment.train_script,
        output_root=tmp_path / "training",
        scale="s55",
        seed=6071406,
        context=frozen_environment.context,
        launch_nonce="a" * 64,
        trainer_sha256=frozen_environment.trainer.sha256,
        device_index=1,
        device_routing_identity=matrix.execution_environment.selected_device_routing_identity(
            frozen_environment.execution_environment
        ),
    )

    assert _command_value(command, "--device") == "cuda:1"


def test_external_trust_root_permissions_location_entropy_and_sealed_transport(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "key"
    key_path.write_bytes(bytes(range(32)))
    key_path.chmod(0o600)
    trust = attestation.load_trust_root(
        key_path,
        repository_root=matrix.REPOSITORY_ROOT,
    )
    assert "<redacted>" in repr(trust) and trust.key.hex() not in repr(trust)

    descriptor = attestation.create_sealed_key_fd(trust)
    inherited = attestation.trust_root_from_sealed_fd(
        descriptor,
        expected_key_id=trust.key_id,
    )
    assert inherited == trust
    with pytest.raises(ValueError, match="does not match its key material"):
        attestation.TrustRoot(key=trust.key[::-1], key_id=trust.key_id)

    key_path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        attestation.load_trust_root(key_path, repository_root=matrix.REPOSITORY_ROOT)
    key_path.chmod(0o600)
    key_path.write_bytes(b"x" * 32)
    with pytest.raises(ValueError, match="low-entropy"):
        attestation.load_trust_root(key_path, repository_root=matrix.REPOSITORY_ROOT)

    repository_key = matrix.REPOSITORY_ROOT / ".test-attestation-key"
    repository_key.write_bytes(bytes(range(32)))
    repository_key.chmod(0o600)
    try:
        with pytest.raises(ValueError, match="outside repository"):
            attestation.load_trust_root(
                repository_key,
                repository_root=matrix.REPOSITORY_ROOT,
            )
    finally:
        repository_key.unlink()


def test_valid_summary_accepts_but_unauthenticated_and_self_rehashed_forgery_fail(
    tmp_path: Path, frozen_environment: FrozenEnvironment
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    nonce = "2" * 64
    summary_path, payload = _write_valid_summary(
        output_root=output_root,
        env=env,
        scale="s55",
        seed=6071406,
        launch_nonce=nonce,
    )
    loaded = matrix.load_validated_training_summary(
        summary_path,
        output_root=output_root,
        train_script=env.train_script,
        context=env.context,
        scale="s55",
        seed=6071406,
        trust_root=env.trust_root,
        launch_nonce=nonce,
        trainer_sha256=env.trainer.sha256,
    )
    assert loaded == payload

    unauthenticated = copy.deepcopy(payload)
    unauthenticated.pop("attestation")
    with pytest.raises(ValueError, match="attestation is missing"):
        _validate(unauthenticated, output_root=output_root, env=env, launch_nonce=nonce)

    forged = copy.deepcopy(payload)
    original_envelope = forged.pop("attestation")
    forged["steps_completed"] = 999
    forged = matrix._digest_bound_payload(forged)
    forged["attestation"] = original_envelope
    with pytest.raises(ValueError, match="Attestation payload checksum|MAC"):
        _validate(forged, output_root=output_root, env=env, launch_nonce=nonce)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_training_summary_top_level_schema_is_exact(
    tmp_path: Path,
    frozen_environment: FrozenEnvironment,
    mutation: str,
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    nonce = "9" * 64
    _, payload = _write_valid_summary(
        output_root=output_root,
        env=env,
        scale="s55",
        seed=6071406,
        launch_nonce=nonce,
    )
    semantic = copy.deepcopy(payload)
    semantic.pop("attestation")
    semantic.pop("payload_sha256")
    if mutation == "missing":
        semantic.pop("task_config")
    else:
        semantic["unregistered"] = True
    mutated = matrix._attested_payload(
        semantic,
        trust_root=env.trust_root,
        purpose=matrix.SUMMARY_ATTESTATION_PURPOSE,
    )

    with pytest.raises(ValueError, match="top-level schema drifted"):
        _validate(mutated, output_root=output_root, env=env, launch_nonce=nonce)


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong-device"])
def test_training_runtime_schema_and_selected_device_semantics_are_exact(
    tmp_path: Path,
    frozen_environment: FrozenEnvironment,
    mutation: str,
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    nonce = "8" * 64
    _, payload = _write_valid_summary(
        output_root=output_root,
        env=env,
        scale="s55",
        seed=6071406,
        launch_nonce=nonce,
    )
    semantic = copy.deepcopy(payload)
    semantic.pop("attestation")
    semantic.pop("payload_sha256")
    runtime = semantic["runtime"]
    if mutation == "missing":
        runtime.pop("device")
    elif mutation == "extra":
        runtime["unregistered"] = True
    else:
        runtime["device"] = "wrong GPU"
    mutated = matrix._attested_payload(
        semantic,
        trust_root=env.trust_root,
        purpose=matrix.SUMMARY_ATTESTATION_PURPOSE,
    )

    expected = "runtime schema drifted" if mutation != "wrong-device" else "selected-device"
    with pytest.raises(ValueError, match=expected):
        _validate(mutated, output_root=output_root, env=env, launch_nonce=nonce)


def test_wrong_key_nonce_trainer_and_checkpoint_substitution_are_rejected(
    tmp_path: Path, frozen_environment: FrozenEnvironment
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    nonce = "3" * 64
    _, payload = _write_valid_summary(
        output_root=output_root,
        env=env,
        scale="s55",
        seed=6071406,
        launch_nonce=nonce,
    )
    wrong_key = attestation._validated_key(bytes(range(32, 64)))
    with pytest.raises(ValueError, match="trust root"):
        _validate(
            payload,
            output_root=output_root,
            env=env,
            launch_nonce=nonce,
            trust_root=wrong_key,
        )
    with pytest.raises(ValueError, match="manifest/implementation binding"):
        _validate(payload, output_root=output_root, env=env, launch_nonce="4" * 64)
    with pytest.raises(ValueError, match="different trainer"):
        _validate(
            payload,
            output_root=output_root,
            env=env,
            launch_nonce=nonce,
            train_script=tmp_path / "arbitrary-trainer.py",
        )

    checkpoint_path = Path(payload["checkpoint"]["path"])
    checkpoint_path.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="byte count|SHA-256"):
        _validate(payload, output_root=output_root, env=env, launch_nonce=nonce)


def test_transcript_optimizer_and_symlink_drift_fail_closed(
    tmp_path: Path, frozen_environment: FrozenEnvironment
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    nonce = "5" * 64
    _, payload = _write_valid_summary(
        output_root=output_root,
        env=env,
        scale="s55",
        seed=6071406,
        launch_nonce=nonce,
    )

    transcript_drift = copy.deepcopy(payload)
    transcript_drift.pop("attestation")
    transcript_drift["training_transcript"]["entries"][500]["loss"] = 0.25
    transcript_drift = matrix._attested_payload(
        transcript_drift,
        trust_root=env.trust_root,
        purpose=matrix.SUMMARY_ATTESTATION_PURPOSE,
    )
    with pytest.raises(ValueError, match="hash chain"):
        _validate(transcript_drift, output_root=output_root, env=env, launch_nonce=nonce)

    checkpoint_path = Path(payload["checkpoint"]["path"])
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    first_state = next(iter(raw["optimizer"]["state"].values()))
    first_state["step"] = torch.tensor(999.0)
    torch.save(raw, checkpoint_path)
    optimizer_drift = copy.deepcopy(payload)
    optimizer_drift.pop("attestation")
    optimizer_drift["checkpoint"] = {
        "path": str(checkpoint_path),
        "sha256": matrix._sha256(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
    }
    optimizer_drift = matrix._attested_payload(
        optimizer_drift,
        trust_root=env.trust_root,
        purpose=matrix.SUMMARY_ATTESTATION_PURPOSE,
    )
    with pytest.raises(ValueError, match="frozen config"):
        _validate(optimizer_drift, output_root=output_root, env=env, launch_nonce=nonce)

    checkpoint_path.unlink()
    checkpoint_path.symlink_to(tmp_path / "missing-checkpoint")
    with pytest.raises(ValueError, match="symbolic link"):
        _validate(payload, output_root=output_root, env=env, launch_nonce=nonce)


def test_atomic_publication_is_exclusive_and_binds_published_bytes(tmp_path: Path) -> None:
    config = _tiny_config("s55")
    model = matrix.trainer.DeepSeekV4ForCausalLM(config)
    probe = matrix.trainer.CSAProbeObjective(
        head_dim=config.head_dim,
        query_dim=config.q_lora_rank,
        vocab_size=config.vocab_size,
    )
    parameters = [*model.parameters(), *probe.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=0.01, fused=True)
    checkpoint_path = tmp_path / "checkpoint.pt"
    metadata = matrix.trainer._save_checkpoint_atomically(
        model,
        probe,
        optimizer,
        config,
        checkpoint_path,
        provenance={"terminal": True},
    )
    assert metadata["sha256"] == matrix._sha256(checkpoint_path)
    assert metadata["bytes"] == checkpoint_path.stat().st_size
    with pytest.raises(ValueError, match="overwrite"):
        matrix.trainer._save_checkpoint_atomically(
            model,
            probe,
            optimizer,
            config,
            checkpoint_path,
            provenance={"terminal": True},
        )

    summary_path = tmp_path / "summary.json"
    matrix.trainer._atomic_write_json(summary_path, {"terminal": True})
    with pytest.raises(ValueError, match="overwrite"):
        matrix.trainer._atomic_write_json(summary_path, {"terminal": False})


def test_trainer_receives_key_only_by_sealed_inherited_fd(
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    monkeypatch.setenv(attestation.KEY_PATH_ENV, str(env.key_path))
    observed: dict[str, Any] = {}

    def fake_subprocess_run(
        command: list[str],
        *,
        check: bool,
        pass_fds: tuple[int, ...],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, check=check, pass_fds=pass_fds, env=env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix.subprocess, "run", fake_subprocess_run)
    lease = FakeGPULease(path=Path("/tmp/test-gpu.lock"), events=[])
    device_guard = FakeGPULease(
        path=matrix.gpu_lock.canonical_device_guard_path(
            matrix.execution_environment.selected_device_routing_identity(env.execution_environment)
        ),
        events=[],
        file_descriptor=3,
    )
    command = [sys.executable, str(env.train_script), "--scale", "s55"]
    result = matrix._run_trainer_from_stable_script(
        command,
        canonical=env.train_script.resolve(),
        expected=env.trainer,
        trust_root=env.trust_root,
        gpu_lease=lease,
        device_guard=device_guard,
    )

    assert result.returncode == 0
    assert observed["check"] is False
    assert len(observed["pass_fds"]) == 4
    assert lease.fileno() in observed["pass_fds"]
    assert device_guard.fileno() in observed["pass_fds"]
    assert observed["command"][1:3] == ["-I", "-c"]
    assert attestation.KEY_FD_ENV in observed["env"]
    assert attestation.KEY_PATH_ENV not in observed["env"]
    assert env.trust_root.key.hex() not in " ".join(observed["command"])


def test_matrix_runs_all_cells_hmac_attests_and_resumes_without_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
    fake_gpu_lease: FakeGPUController,
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    matrix_summary = output_root / matrix.MATRIX_SUMMARY.name
    commands: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        canonical: Path,
        expected: matrix.CanonicalTrainerSnapshot,
        trust_root: attestation.TrustRoot,
        gpu_lease: FakeGPULease,
        device_guard: FakeGPULease,
    ) -> subprocess.CompletedProcess[str]:
        assert fake_gpu_lease.leases
        assert not fake_gpu_lease.leases[-1].closed
        assert canonical == env.train_script.resolve()
        assert expected == env.trainer
        assert trust_root == env.trust_root
        gpu_lease.assert_held()
        device_guard.assert_held()
        commands.append(command)
        scale = _command_value(command, "--scale")
        seed = int(_command_value(command, "--seed"))
        _write_valid_summary(
            output_root=output_root,
            env=env,
            scale=scale,
            seed=seed,
            launch_nonce=_command_value(command, "--launch-nonce"),
            command=command,
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix, "_run_trainer_from_stable_script", fake_run)
    terminal = matrix.run_matrix(
        manifest_path=env.manifest_path,
        output_root=output_root,
        matrix_summary=matrix_summary,
        train_script=env.train_script,
        attestation_key_path=env.key_path,
    )
    assert len(commands) == 10
    assert terminal["status"] == "terminal"
    assert terminal["completed_runs"] == terminal["expected_runs"] == 10
    assert "inode" not in terminal["gpu_lease"]
    assert terminal["gpu_lease"]["child_nested_lease"] is False
    assert terminal["gpu_lease"]["selected_device_routing_identity"] == {
        "identity_type": "uuid",
        "identity": "GPU-test-0",
    }
    assert terminal["gpu_lease"]["device_guard"]["semantics"] == (
        matrix.gpu_lock.DEVICE_GUARD_SEMANTICS
    )
    assert len({record["launch_nonce"] for record in terminal["runs"]}) == 10
    assert len(fake_gpu_lease.leases) == 2
    assert all(lease.closed for lease in fake_gpu_lease.leases)
    assert fake_gpu_lease.events[0].startswith("acquire:p2-direct-training-matrix:")
    semantic = dict(terminal)
    envelope = semantic.pop("attestation")
    attestation.verify_attestation(
        semantic,
        envelope,
        trust_root=env.trust_root,
        purpose=matrix.MATRIX_ATTESTATION_PURPOSE,
    )

    resumed = matrix.run_matrix(
        manifest_path=env.manifest_path,
        output_root=output_root,
        matrix_summary=matrix_summary,
        train_script=env.train_script,
        attestation_key_path=env.key_path,
    )
    assert resumed == terminal
    assert len(commands) == 10
    assert len(fake_gpu_lease.leases) == 4
    assert fake_gpu_lease.leases[-1].closed

    for mutation in ("missing", "extra"):
        schema_tamper = copy.deepcopy(terminal)
        schema_tamper.pop("attestation")
        schema_tamper.pop("payload_sha256")
        if mutation == "missing":
            schema_tamper.pop("minimum_steps")
        else:
            schema_tamper["unregistered"] = True
        schema_tamper = matrix._attested_payload(
            schema_tamper,
            trust_root=env.trust_root,
            purpose=matrix.MATRIX_ATTESTATION_PURPOSE,
        )
        matrix_summary.write_text(json.dumps(schema_tamper), encoding="utf-8")
        with pytest.raises(ValueError, match="top-level schema drifted"):
            matrix.run_matrix(
                manifest_path=env.manifest_path,
                output_root=output_root,
                matrix_summary=matrix_summary,
                train_script=env.train_script,
                attestation_key_path=env.key_path,
            )
        matrix_summary.write_text(json.dumps(terminal), encoding="utf-8")

    environment_tamper = copy.deepcopy(terminal)
    environment_tamper.pop("attestation")
    environment_tamper.pop("payload_sha256")
    environment_tamper["execution_environment"]["cuda_driver_version"] = "571.00"
    environment_tamper = matrix._attested_payload(
        environment_tamper,
        trust_root=env.trust_root,
        purpose=matrix.MATRIX_ATTESTATION_PURPOSE,
    )
    matrix_summary.write_text(json.dumps(environment_tamper), encoding="utf-8")
    with pytest.raises(ValueError, match="changed on exact resume"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=output_root,
            matrix_summary=matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )
    matrix_summary.write_text(json.dumps(terminal), encoding="utf-8")

    rogue = output_root / "unregistered-after-terminal.txt"
    rogue.write_text("rogue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unregistered orphan"):
        matrix.load_terminal_matrix_record(
            matrix_summary,
            context=env.context,
            trust_root=env.trust_root,
            trainer_binding=env.trainer.public_binding,
            scale="s55",
            seed=6071406,
        )
    rogue.unlink()

    lease_semantic_tamper = copy.deepcopy(terminal)
    lease_semantic_tamper.pop("attestation")
    lease_semantic_tamper.pop("payload_sha256")
    lease_semantic_tamper["gpu_lease"]["child_nested_lease"] = True
    lease_semantic_tamper = matrix._attested_payload(
        lease_semantic_tamper,
        trust_root=env.trust_root,
        purpose=matrix.MATRIX_ATTESTATION_PURPOSE,
    )
    matrix_summary.write_text(json.dumps(lease_semantic_tamper), encoding="utf-8")
    with pytest.raises(ValueError, match="GPU lease semantics drifted"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=output_root,
            matrix_summary=matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )
    assert fake_gpu_lease.leases[-1].closed
    matrix_summary.write_text(json.dumps(terminal), encoding="utf-8")

    with pytest.raises(ValueError, match="GPU lease path or semantics drifted"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=output_root,
            matrix_summary=matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
            gpu_lock_path=tmp_path / "different-device.lock",
        )
    assert len(commands) == 10
    assert fake_gpu_lease.leases[-1].closed

    tampered = copy.deepcopy(terminal)
    tampered["steps"] = 999
    tampered = matrix._digest_bound_payload(tampered)
    tampered["attestation"] = terminal["attestation"]
    matrix_summary.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="Attestation payload checksum|MAC"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=output_root,
            matrix_summary=matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )
    assert len(commands) == 10


def test_strict_admission_promotes_first_cell_validates_ledger_then_runs_exactly_nine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
    fake_gpu_lease: FakeGPUController,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    evidence_paths = (
        case.superseded_manifest,
        case.superseded_matrix,
        case.preserved_claim,
        case.training_summary,
        case.checkpoint,
    )
    evidence_before = {path: matrix._sha256(path) for path in evidence_paths}
    commands: list[list[str]] = []
    events: list[str] = []
    original_validate = matrix.validate_matrix_summary

    def observe_validation(payload: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        result = original_validate(payload, **kwargs)
        events.append(f"validated:{payload['completed_runs']}")
        return result

    def fake_run(
        command: list[str],
        *,
        canonical: Path,
        expected: matrix.CanonicalTrainerSnapshot,
        trust_root: attestation.TrustRoot,
        gpu_lease: FakeGPULease,
        device_guard: FakeGPULease,
    ) -> subprocess.CompletedProcess[str]:
        assert canonical == env.train_script.resolve()
        assert expected == env.trainer
        assert trust_root == env.trust_root
        gpu_lease.assert_held()
        device_guard.assert_held()
        scale = _command_value(command, "--scale")
        seed = int(_command_value(command, "--seed"))
        events.append(f"child:{scale}/{seed}")
        commands.append(command)
        _write_valid_summary(
            output_root=case.output_root,
            env=env,
            scale=scale,
            seed=seed,
            launch_nonce=_command_value(command, "--launch-nonce"),
            command=command,
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix, "validate_matrix_summary", observe_validation)
    monkeypatch.setattr(matrix, "_run_trainer_from_stable_script", fake_run)
    terminal = matrix.run_matrix(
        manifest_path=env.manifest_path,
        output_root=case.output_root,
        matrix_summary=case.current_matrix_summary,
        train_script=env.train_script,
        attestation_key_path=env.key_path,
    )

    assert terminal["status"] == "terminal"
    assert terminal["completed_runs"] == 10
    assert len(commands) == 9
    assert (
        _command_value(commands[0], "--scale"),
        int(_command_value(commands[0], "--seed")),
    ) == ("s55", 6071407)
    first_child_index = next(
        index for index, event in enumerate(events) if event.startswith("child:")
    )
    assert "validated:1" in events[:first_child_index]
    assert (terminal["runs"][0]["scale"], terminal["runs"][0]["seed"]) == (
        "s55",
        6071406,
    )
    assert terminal["preheldout_admission"] is not None
    assert {path: matrix._sha256(path) for path in evidence_paths} == evidence_before
    assert all(lease.closed for lease in fake_gpu_lease.leases)

    ledger, first_record = matrix.load_terminal_matrix_record(
        case.current_matrix_summary,
        context=env.context,
        trust_root=env.trust_root,
        trainer_binding=env.trainer.public_binding,
        scale="s55",
        seed=6071406,
    )
    assert ledger["status"] == "terminal"
    assert (first_record["scale"], first_record["seed"]) == ("s55", 6071406)
    summary, checkpoint, raw_checkpoint = matrix.load_validated_training_bundle_for_ledger_record(
        case.training_summary,
        output_root=case.output_root.resolve(),
        context=env.context,
        scale="s55",
        seed=6071406,
        trust_root=env.trust_root,
        trainer_binding=env.trainer.public_binding,
        ledger_record=first_record,
        expected_execution_environment=env.execution_environment,
    )
    assert summary["seed"] == 6071406
    assert checkpoint["sha256"] == matrix._sha256(case.checkpoint)
    assert raw_checkpoint["provenance"]["training_seed"] == 6071406
    assert raw_checkpoint["checkpoint_schema_version"] == (
        matrix.trainer.DIRECT_CHECKPOINT_SCHEMA_VERSION
    )


def test_amended_ledger_replacement_after_validation_blocks_first_new_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    original_snapshot = matrix._open_exact_matrix_snapshot
    child_calls = 0

    def replace_after_snapshot(
        path: Path, payload: dict[str, Any]
    ) -> attestation.OpenedRegularFile:
        opened = original_snapshot(path, payload)
        if payload["completed_runs"] == 1:
            replacement = path.with_name("replacement-ledger.json")
            replacement.write_text("{}\n", encoding="utf-8")
            replacement.chmod(0o600)
            matrix.os.replace(replacement, path)
        return opened

    def forbidden_child(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal child_calls
        child_calls += 1
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(matrix, "_open_exact_matrix_snapshot", replace_after_snapshot)
    monkeypatch.setattr(matrix, "_run_trainer_from_stable_script", forbidden_child)
    with pytest.raises(ValueError, match="changed during validation|replaced during validation"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )
    assert child_calls == 0


def test_admission_replacement_after_ledger_validation_blocks_first_new_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    original_snapshot = matrix._open_exact_admission_snapshot
    child_calls = 0

    def replace_after_snapshot(
        binding: dict[str, Any],
    ) -> attestation.OpenedRegularFile:
        opened = original_snapshot(binding)
        replacement = opened.path.with_name("replacement-admission-before-child.json")
        replacement.write_text("{}\n", encoding="utf-8")
        replacement.chmod(0o600)
        matrix.os.replace(replacement, opened.path)
        return opened

    def forbidden_child(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal child_calls
        child_calls += 1
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(matrix, "_open_exact_admission_snapshot", replace_after_snapshot)
    monkeypatch.setattr(matrix, "_run_trainer_from_stable_script", forbidden_child)
    with pytest.raises(ValueError, match="changed during validation|replaced during validation"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )
    assert child_calls == 0


@pytest.mark.parametrize(
    ("blocker_name", "as_directory", "message"),
    [
        ("rogue-quality.json", False, "unregistered direct artifact"),
        ("calibration", True, "downstream artifact"),
    ],
)
def test_incomplete_admitted_resume_blocks_direct_root_contamination_before_child(
    blocker_name: str,
    as_directory: bool,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    case = _prepare_preheldout_admission_case(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        env=env,
    )
    original_validate = matrix.validate_matrix_summary
    stopped = False

    def stop_after_initial_ledger(payload: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        nonlocal stopped
        result = original_validate(payload, **kwargs)
        if not stopped and payload["completed_runs"] == 1:
            stopped = True
            raise RuntimeError("injected stop after amended ledger")
        return result

    monkeypatch.setattr(matrix, "validate_matrix_summary", stop_after_initial_ledger)
    with pytest.raises(RuntimeError, match="injected stop"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )
    assert case.current_matrix_summary.is_file()
    assert (
        json.loads(case.current_matrix_summary.read_text(encoding="utf-8"))["completed_runs"] == 1
    )

    blocker = case.output_root.parent / blocker_name
    if as_directory:
        blocker.mkdir()
    else:
        blocker.write_text("{}\n", encoding="utf-8")
    child_calls = 0

    def forbidden_child(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal child_calls
        child_calls += 1
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(matrix, "validate_matrix_summary", original_validate)
    monkeypatch.setattr(matrix, "_run_trainer_from_stable_script", forbidden_child)
    with pytest.raises(ValueError, match=message):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=case.output_root,
            matrix_summary=case.current_matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )
    assert child_calls == 0


def test_final_training_child_environment_drift_blocks_terminal_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    matrix_summary = output_root / matrix.MATRIX_SUMMARY.name
    returned_children = 0

    def fake_run(
        command: list[str],
        **_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal returned_children
        scale = _command_value(command, "--scale")
        seed = int(_command_value(command, "--seed"))
        _write_valid_summary(
            output_root=output_root,
            env=env,
            scale=scale,
            seed=seed,
            launch_nonce=_command_value(command, "--launch-nonce"),
            command=command,
        )
        returned_children += 1
        return subprocess.CompletedProcess(command, 0)

    def assert_exact(_expected: dict[str, Any]) -> None:
        if returned_children == len(matrix.FROZEN_SCALES) * len(matrix.FROZEN_TRAINING_SEEDS):
            raise ValueError("Exact execution environment changed after final training child.")

    monkeypatch.setattr(matrix, "_run_trainer_from_stable_script", fake_run)
    monkeypatch.setattr(
        matrix.execution_environment,
        "assert_exact_execution_environment",
        assert_exact,
    )
    with pytest.raises(ValueError, match="changed after final training child"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=output_root,
            matrix_summary=matrix_summary,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )

    ledger = json.loads(matrix_summary.read_text(encoding="utf-8"))
    assert ledger["status"] == "in_progress"
    assert ledger["completed_runs"] == 9
    final_scale = matrix.FROZEN_SCALES[-1]
    final_seed = matrix.FROZEN_TRAINING_SEEDS[-1]
    final_output = matrix._run_output_dir(output_root, final_scale, final_seed)
    assert matrix._training_summary_path(output_root, final_scale, final_seed).is_file()
    assert (final_output / matrix.CLAIM_NAME).is_file()


def test_gpu_lease_closes_when_canonical_path_validation_fails(
    tmp_path: Path,
    frozen_environment: FrozenEnvironment,
    fake_gpu_lease: FakeGPUController,
) -> None:
    target = tmp_path / "gpu-device.lock"
    alias = tmp_path / "gpu-device.alias"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        matrix.run_matrix(
            manifest_path=frozen_environment.manifest_path,
            output_root=tmp_path / "training",
            matrix_summary=tmp_path / "training" / matrix.MATRIX_SUMMARY.name,
            train_script=frozen_environment.train_script,
            attestation_key_path=frozen_environment.key_path,
            gpu_lock_path=alias,
        )

    assert len(fake_gpu_lease.leases) == 2
    assert all(lease.closed for lease in fake_gpu_lease.leases)


def test_arbitrary_trainer_stale_claim_and_concurrent_runner_are_rejected(
    tmp_path: Path, frozen_environment: FrozenEnvironment
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    arbitrary = tmp_path / "arbitrary.py"
    arbitrary.write_text("raise SystemExit(0)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=output_root,
            matrix_summary=output_root / matrix.MATRIX_SUMMARY.name,
            train_script=arbitrary,
            attestation_key_path=env.key_path,
        )

    stale_dir = output_root / "s55" / "seed-6071406"
    stale_dir.mkdir(parents=True)
    (stale_dir / matrix.CLAIM_NAME).write_text("stale\n", encoding="utf-8")
    with pytest.raises(ValueError, match="claim"):
        matrix.run_matrix(
            manifest_path=env.manifest_path,
            output_root=output_root,
            matrix_summary=output_root / matrix.MATRIX_SUMMARY.name,
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )

    (stale_dir / matrix.CLAIM_NAME).unlink()
    with matrix._matrix_lock(output_root):
        with pytest.raises(ValueError, match="exclusive lock"):
            matrix.run_matrix(
                manifest_path=env.manifest_path,
                output_root=output_root,
                matrix_summary=output_root / matrix.MATRIX_SUMMARY.name,
                train_script=env.train_script,
                attestation_key_path=env.key_path,
            )


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("symlink", "opened safely"),
        ("hardlink", "link count"),
        ("unsafe-mode", "mode is unsafe"),
    ],
)
def test_matrix_lock_rejects_unsafe_preexisting_files(
    tmp_path: Path,
    kind: str,
    message: str,
) -> None:
    output_root = tmp_path / "training"
    lock_path = output_root.parent / matrix.LOCK_NAME
    target = tmp_path / "lock-target"
    target.write_bytes(b"preexisting\n")
    target.chmod(0o600)
    if kind == "symlink":
        lock_path.symlink_to(target)
    elif kind == "hardlink":
        matrix.os.link(target, lock_path)
    else:
        lock_path.write_bytes(b"unsafe\n")
        lock_path.chmod(0o644)

    with pytest.raises(ValueError, match=message):
        with matrix._matrix_lock(output_root):
            raise AssertionError("unsafe training lock unexpectedly acquired")


@pytest.mark.parametrize("mutation", ["delete", "replace"])
def test_lock_deletion_or_replacement_during_child_fails_before_prefix_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
    mutation: str,
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    matrix_summary = output_root / matrix.MATRIX_SUMMARY.name
    lock_path = output_root.parent / matrix.LOCK_NAME
    replacement_descriptors: list[int] = []

    def tamper_during_child(
        command: list[str],
        *,
        canonical: Path,
        expected: matrix.CanonicalTrainerSnapshot,
        trust_root: attestation.TrustRoot,
        gpu_lease: FakeGPULease,
        device_guard: FakeGPULease,
    ) -> subprocess.CompletedProcess[str]:
        assert canonical == env.train_script.resolve()
        assert expected == env.trainer
        assert trust_root == env.trust_root
        gpu_lease.assert_held()
        device_guard.assert_held()
        lock_path.unlink()
        if mutation == "replace":
            lock_path.write_bytes(b"replacement-lock\n")
            lock_path.chmod(0o600)
            descriptor = matrix.os.open(
                lock_path,
                matrix.os.O_RDWR | getattr(matrix.os, "O_CLOEXEC", 0),
            )
            matrix.fcntl.flock(
                descriptor,
                matrix.fcntl.LOCK_EX | matrix.fcntl.LOCK_NB,
            )
            replacement_descriptors.append(descriptor)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix, "_run_trainer_from_stable_script", tamper_during_child)
    try:
        with pytest.raises(ValueError, match="deleted|replaced"):
            matrix.run_matrix(
                manifest_path=env.manifest_path,
                output_root=output_root,
                matrix_summary=matrix_summary,
                train_script=env.train_script,
                attestation_key_path=env.key_path,
            )
    finally:
        for descriptor in replacement_descriptors:
            matrix.fcntl.flock(descriptor, matrix.fcntl.LOCK_UN)
            matrix.os.close(descriptor)

    partial = json.loads(matrix_summary.read_text(encoding="utf-8"))
    assert partial["status"] == "in_progress"
    assert partial["completed_runs"] == 0
    assert partial["runs"] == []
    assert bool(replacement_descriptors) is (mutation == "replace")
