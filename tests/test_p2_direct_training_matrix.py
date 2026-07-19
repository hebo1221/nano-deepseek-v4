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


@dataclass(frozen=True)
class FrozenEnvironment:
    manifest_path: Path
    train_script: Path
    key_path: Path
    trust_root: attestation.TrustRoot
    context: matrix.FrozenContext
    trainer: matrix.CanonicalTrainerSnapshot


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
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": matrix.EXPERIMENT_ID,
        "scale": scale,
        "seed": seed,
        **matrix._seed_values(seed),
        "source": env.context.source,
        "config": matrix.asdict(config),
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
    )

    assert matrix.FROZEN_SCALES == ("s55", "s151")
    assert matrix.FROZEN_TRAINING_SEEDS == tuple(range(6071406, 6071411))
    assert matrix.STEPS == matrix.MINIMUM_STEPS == 1_000
    assert _command_value(command, "--steps") == "1000"
    assert _command_value(command, "--minimum-steps") == "1000"
    assert "--disable-early-stop" in command
    assert _command_value(command, "--launch-nonce") == nonce
    assert json.loads(_command_value(command, "--direct-hyperparameters-json")) == (
        matrix.FROZEN_HYPERPARAMETERS
    )


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
        pass_fds: tuple[int, int],
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        observed.update(command=command, check=check, pass_fds=pass_fds, env=env)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(matrix.subprocess, "run", fake_subprocess_run)
    command = [sys.executable, str(env.train_script), "--scale", "s55"]
    result = matrix._run_trainer_from_stable_script(
        command,
        canonical=env.train_script.resolve(),
        expected=env.trainer,
        trust_root=env.trust_root,
    )

    assert result.returncode == 0
    assert observed["check"] is False
    assert len(observed["pass_fds"]) == 2
    assert attestation.KEY_FD_ENV in observed["env"]
    assert attestation.KEY_PATH_ENV not in observed["env"]
    assert env.trust_root.key.hex() not in " ".join(observed["command"])


def test_matrix_runs_all_cells_hmac_attests_and_resumes_without_relaunch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_environment: FrozenEnvironment,
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
    ) -> subprocess.CompletedProcess[str]:
        assert canonical == env.train_script.resolve()
        assert expected == env.trainer
        assert trust_root == env.trust_root
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
    assert len({record["launch_nonce"] for record in terminal["runs"]}) == 10
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
