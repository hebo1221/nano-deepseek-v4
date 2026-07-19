from __future__ import annotations

import copy
import json
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


@pytest.fixture(autouse=True)
def fake_gpu_lease(monkeypatch: pytest.MonkeyPatch) -> FakeGPUController:
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
    matrix_summary = output_root / "training-matrix.summary.json"
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


def test_final_training_child_environment_drift_blocks_terminal_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
) -> None:
    env = frozen_environment
    output_root = tmp_path / "training"
    matrix_summary = output_root / "training-matrix.summary.json"
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
            matrix_summary=tmp_path / "training" / "training-matrix.summary.json",
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
            matrix_summary=output_root / "training-matrix.summary.json",
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
            matrix_summary=output_root / "training-matrix.summary.json",
            train_script=env.train_script,
            attestation_key_path=env.key_path,
        )

    (stale_dir / matrix.CLAIM_NAME).unlink()
    with matrix._matrix_lock(output_root):
        with pytest.raises(ValueError, match="exclusive lock"):
            matrix.run_matrix(
                manifest_path=env.manifest_path,
                output_root=output_root,
                matrix_summary=output_root / "training-matrix.summary.json",
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
    matrix_summary = output_root / "training-matrix.summary.json"
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
