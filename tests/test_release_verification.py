from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.verify_pypi_release import (
    ReleaseVerificationError,
    load_release_manifest,
    verify_release_payload,
    wait_for_release,
)

WHEEL = "nano_deepseek_v4-0.3.0-py3-none-any.whl"
SDIST = "nano_deepseek_v4-0.3.0.tar.gz"
WHEEL_SHA = "a" * 64
SDIST_SHA = "b" * 64
ROOT = Path(__file__).parents[1]
README_DOCS_PREFIX = "https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/"
README_LOCAL_FILE_PREFIXES = {
    README_DOCS_PREFIX: ROOT / "docs",
    "https://github.com/hebo1221/nano-deepseek-v4/blob/main/nano_deepseek_v4/": (
        ROOT / "nano_deepseek_v4"
    ),
    "https://github.com/hebo1221/nano-deepseek-v4/blob/main/references/": (
        ROOT / "references"
    ),
}


def _workflow_jobs(filename: str) -> dict[str, Any]:
    payload = yaml.safe_load(
        (ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    jobs = payload.get("jobs")
    assert isinstance(jobs, dict)
    return jobs


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    steps = job.get("steps")
    assert isinstance(steps, list)
    matches = [step for step in steps if isinstance(step, dict) and step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repository": "hebo1221/nano-deepseek-v4",
        "tag": "v0.3.0",
        "commit": "c" * 40,
        "version": "0.3.0",
        "artifacts": {WHEEL: WHEEL_SHA, SDIST: SDIST_SHA},
    }


def _payload() -> dict[str, Any]:
    return {
        "info": {"name": "nano-deepseek-v4", "version": "0.3.0"},
        "urls": [
            {"filename": SDIST, "digests": {"sha256": SDIST_SHA}},
            {"filename": WHEEL, "digests": {"sha256": WHEEL_SHA}},
        ],
    }


def test_release_payload_matches_exact_artifact_inventory():
    published = verify_release_payload("nano_deepseek_v4", _manifest(), _payload())

    assert published == {SDIST: SDIST_SHA, WHEEL: WHEEL_SHA}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["urls"].pop(), "missing="),
        (
            lambda payload: payload["urls"].append(
                {"filename": "unexpected.zip", "digests": {"sha256": "d" * 64}}
            ),
            "unexpected=",
        ),
        (
            lambda payload: payload["urls"][0]["digests"].update(sha256="e" * 64),
            "sha256_mismatch=",
        ),
    ],
)
def test_release_payload_rejects_inventory_drift(mutate, message: str):
    payload = _payload()
    mutate(payload)

    with pytest.raises(ReleaseVerificationError, match=message):
        verify_release_payload("nano-deepseek-v4", _manifest(), payload)


def test_release_payload_rejects_wrong_project_or_version():
    wrong_project = _payload()
    wrong_project["info"]["name"] = "another-project"
    with pytest.raises(ReleaseVerificationError, match="project name"):
        verify_release_payload("nano-deepseek-v4", _manifest(), wrong_project)

    wrong_version = _payload()
    wrong_version["info"]["version"] = "0.2.0"
    with pytest.raises(ReleaseVerificationError, match="version"):
        verify_release_payload("nano-deepseek-v4", _manifest(), wrong_version)


def test_wait_for_release_retries_partial_index_until_complete():
    partial = _payload()
    partial["urls"].pop()
    responses = iter([partial, _payload()])
    calls: list[tuple[str, float]] = []
    sleeps: list[float] = []

    def fetcher(url: str, timeout: float):
        calls.append((url, timeout))
        return next(responses)

    published = wait_for_release(
        "nano-deepseek-v4",
        _manifest(),
        attempts=2,
        delay_seconds=0.25,
        timeout_seconds=3.0,
        fetcher=fetcher,
        sleeper=sleeps.append,
    )

    assert published[WHEEL] == WHEEL_SHA
    assert calls == [
        ("https://pypi.org/pypi/nano-deepseek-v4/0.3.0/json", 3.0),
        ("https://pypi.org/pypi/nano-deepseek-v4/0.3.0/json", 3.0),
    ]
    assert sleeps == [0.25]


def test_wait_for_release_reports_last_error_without_extra_sleep():
    def fetcher(_url: str, _timeout: float):
        raise ReleaseVerificationError("not visible")

    sleeps: list[float] = []
    with pytest.raises(ReleaseVerificationError, match="after 2 attempts: not visible"):
        wait_for_release(
            "nano-deepseek-v4",
            _manifest(),
            attempts=2,
            delay_seconds=1.0,
            fetcher=fetcher,
            sleeper=sleeps.append,
        )

    assert sleeps == [1.0]


def test_load_release_manifest_rejects_invalid_artifact_digest(tmp_path: Path):
    path = tmp_path / "release-manifest.json"
    payload = _manifest()
    payload["artifacts"][WHEEL] = "not-a-digest"
    path.write_text(json.dumps(payload))

    with pytest.raises(ReleaseVerificationError, match="invalid SHA-256"):
        load_release_manifest(path)


def test_readme_links_are_safe_outside_the_source_checkout():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme)

    relative = [
        target
        for target in targets
        if not target.startswith(("https://", "http://", "#"))
    ]
    assert relative == []

    docs_targets = [target for target in targets if target.startswith(README_DOCS_PREFIX)]
    assert docs_targets

    for prefix, local_root in README_LOCAL_FILE_PREFIXES.items():
        for target in targets:
            if target.startswith(prefix):
                local_path = local_root / target.removeprefix(prefix)
                assert local_path.is_file(), target


@pytest.mark.parametrize(
    "relative",
    ["README.md", "docs/guides/installation.md"],
)
def test_matching_tag_notebook_snippet_installs_matching_runtime(relative: str):
    document = (ROOT / relative).read_text(encoding="utf-8")
    version = (
        'NANO_VERSION="$(python -c \'from importlib.metadata import version; '
        'print(version("nano-deepseek-v4"))\')"'
    )
    runtime = 'python -m pip install "nano-deepseek-v4[notebook]==${NANO_VERSION}"'
    download = (
        '"https://raw.githubusercontent.com/hebo1221/nano-deepseek-v4/'
        'v${NANO_VERSION}/notebooks/01_flash_inference.ipynb"'
    )
    launch = "jupyter lab 01_flash_inference.ipynb"

    assert document.count(version) == 1
    assert document.count(runtime) == 1
    assert document.count(download) == 1
    assert document.count(launch) == 1
    assert document.index(version) < document.index(runtime) < document.index(
        download
    ) < document.index(launch)


@pytest.mark.parametrize(
    "workflow_path",
    [ROOT / ".github" / "workflows" / "ci.yml", ROOT / ".github" / "workflows" / "release.yml"],
)
def test_built_wheel_smoke_covers_both_official_flash_presets(workflow_path: Path):
    workflow = workflow_path.read_text(encoding="utf-8")
    filename = workflow_path.name
    job_name = "package" if filename == "ci.yml" else "build"
    wheel_step_name = (
        "Smoke-test the built wheel in an isolated environment"
        if filename == "ci.yml"
        else "Smoke-test the release wheel"
    )
    wheel_smoke = _named_step(
        _workflow_jobs(filename)[job_name], wheel_step_name
    )["run"]

    assert "--preset flash \\" in workflow
    assert "--preset flash-0731 \\" in workflow
    assert '"${NANO_WHEEL_PATH}[official]"' in workflow
    assert '"${NANO_WHEEL_PATH}[parity]"' in workflow
    assert '"${NANO_WHEEL_PATH}[notebook]"' in workflow
    assert 'python -c "import huggingface_hub"' in workflow
    assert 'python -c "import transformers"' in workflow
    assert "expected exactly one built wheel" in workflow
    assert "wheel smoke imported source checkout" in workflow
    assert "--override-ini=pythonpath=" in workflow
    assert "--import-mode=importlib" in workflow
    assert '"$GITHUB_WORKSPACE/tests/test_hub_checkpoint.py"' in workflow
    assert '"$GITHUB_WORKSPACE/tests/test_dspark_hub_checkpoint.py"' in workflow
    assert '"$GITHUB_WORKSPACE/tests/test_verify_flash_0731_receipt.py"' in workflow
    assert '"$GITHUB_WORKSPACE/tests/test_conformance.py"' in workflow
    assert '"$GITHUB_WORKSPACE/tests/test_reproduce.py"' in workflow
    assert (
        '"$GITHUB_WORKSPACE/tests/test_core.py::'
        'test_public_version_matches_distribution_metadata"'
        in workflow
    )
    assert '"$GITHUB_WORKSPACE/tests/test_notebook.py"' in workflow
    assert "/bin/jupyter nbconvert" in workflow
    assert "--execute" in workflow
    assert "test -f /tmp/executed-notebook.ipynb" in workflow
    assert "nano-deepseek-v4 = nano_deepseek_v4.cli:main" in workflow
    assert "nano-deepseek-v4-verify-flash-0731 --help" in workflow
    assert "/bin/nano-deepseek-v4 --version" in workflow
    assert "/bin/nano-deepseek-v4 --help" in workflow
    assert "/bin/python -m nano_deepseek_v4 --help" in workflow
    assert "/bin/nano-deepseek-v4 demo" in workflow
    assert "/bin/nano-deepseek-v4 conformance" in workflow
    assert "--profile core" in workflow
    assert "--profile offline" in workflow
    assert 'name == "triton" or name.startswith(("cuda-", "nvidia-"))' in workflow
    assert "DeepSeek-V4-Flash-0731-config.json" in workflow
    assert "DeepSeek-V4-Flash-0731-metadata.json" in workflow
    assert "tiny-text-training-baseline.json" in workflow
    assert "installed sdist reproduction contract differs from source" in workflow
    assert "wheel reproduction contract differs from source" in workflow
    assert "/bin/nano-deepseek-v4-train --help" in workflow
    assert re.search(
        r'"\$GITHUB_WORKSPACE/scripts/run_no_network\.py" \\\n'
        r"\s+/tmp/[^\s]+/bin/nano-deepseek-v4 reproduce \\",
        workflow,
    )
    assert "--save-directory /tmp/nano-deepseek-v4-" in workflow
    assert "--output /tmp/nano-deepseek-v4-" in workflow
    assert 'acceptance.get("passed") is not True' in workflow
    assert 'historical.get("informational_only") is not True' in workflow
    assert 'not historical.get("reference_id")' in workflow
    assert 'historical_status not in {"exact_match", "different"}' in workflow
    assert '(historical_status == "exact_match") != (historical_differences == [])' in workflow
    assert "/bin/nano-deepseek-v4-reproduce" not in workflow
    assert wheel_smoke.count("/bin/nano-deepseek-v4 reproduce") == 1
    assert wheel_smoke.count("/bin/nano-deepseek-v4 generate") == 1
    reproduce_index = wheel_smoke.index("/bin/nano-deepseek-v4 reproduce")
    generate_index = wheel_smoke.index(
        "/bin/nano-deepseek-v4 generate", reproduce_index
    )
    receipt_index = wheel_smoke.index(
        'acceptance.get("passed") is not True', reproduce_index
    )
    official_install_index = wheel_smoke.index('"${NANO_WHEEL_PATH}[official]"')
    inspect_bundle_index = wheel_smoke.index(
        "/bin/nano-deepseek-v4-inspect", official_install_index
    )
    legacy_generate_help_index = wheel_smoke.index(
        "/bin/nano-deepseek-v4-generate --help", inspect_bundle_index
    )
    compare_index = wheel_smoke.index(
        "/bin/nano-deepseek-v4-compare", legacy_generate_help_index
    )
    assert reproduce_index < generate_index < receipt_index < official_install_index
    assert official_install_index < inspect_bundle_index < legacy_generate_help_index
    assert legacy_generate_help_index < compare_index
    assert official_install_index < wheel_smoke.index(
        'python -c "import huggingface_hub"'
    ) < wheel_smoke.index('"${NANO_WHEEL_PATH}[parity]"')


def test_minimum_dependencies_job_is_pinned_and_required():
    jobs = _workflow_jobs("ci.yml")
    minimum = jobs["minimum-dependencies"]

    assert minimum["runs-on"] == "ubuntu-24.04"
    setup_python = next(
        step
        for step in minimum["steps"]
        if isinstance(step, dict) and str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    assert setup_python["with"]["python-version"] == "3.10"

    install = _named_step(minimum, "Install minimum dependencies")["run"]
    assert "--index-url https://download.pytorch.org/whl/cpu" in install
    assert '"torch==2.4.0"' in install
    assert '"numpy==1.24.0"' in install
    assert '"safetensors==0.6.1"' in install
    assert "from packaging.version import Version" in install
    assert 'Version(torch_version).base_version != "2.4.0"' in install
    assert "torch.version.cuda is not None" in install
    assert 'python -m pip install -e ".[dev]"' in install
    assert "python -m pip check" in install
    assert _named_step(minimum, "Run full test suite")["run"].strip() == "pytest"

    required = jobs["required"]
    assert "minimum-dependencies" in required["needs"]
    required_step = _named_step(required, "Require every CI job")
    assert required_step["env"]["MINIMUM_DEPENDENCIES_RESULT"] == (
        "${{ needs.minimum-dependencies.result }}"
    )
    assert '"$MINIMUM_DEPENDENCIES_RESULT"' in required_step["run"]


def test_release_workflow_runs_source_verifier_before_cuda_gate():
    jobs = _workflow_jobs("release.yml")
    build = jobs["build"]
    source = _named_step(build, "Verify release source and version")["run"]
    step_names = [step.get("name") for step in build["steps"]]
    dependencies = _named_step(
        build, "Install release source verifier dependencies"
    )["run"]

    assert '"PyYAML>=6.0"' in dependencies
    assert "git fetch --no-tags origin main:refs/remotes/origin/main" in source
    assert 'git merge-base --is-ancestor "$GITHUB_SHA" origin/main' in source
    assert 'python scripts/verify_release_source.py --tag "$RELEASE_TAG"' in source
    assert "python - <<'PY'" not in source
    assert (
        step_names.index("Install release source verifier dependencies")
        < step_names.index("Verify release source and version")
        < step_names.index("Require successful CI for the release commit")
        < step_names.index("Verify and stage the CUDA release receipt")
    )


def test_release_workflow_requires_successful_ci_for_exact_tag_sha():
    workflow = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    )
    build = workflow["jobs"]["build"]
    command = _named_step(
        build, "Require successful CI for the release commit"
    )["run"]

    assert workflow["permissions"]["actions"] == "read"
    assert workflow["permissions"]["contents"] == "read"
    assert 'GH_TOKEN: ${{ github.token }}' in (
        ROOT / ".github" / "workflows" / "release.yml"
    ).read_text(encoding="utf-8")
    assert "/actions/workflows/ci.yml/runs" in command
    assert '-f head_sha="$GITHUB_SHA"' in command
    assert "-f status=completed" in command
    assert '.event == "push"' in command
    assert '.head_branch == "main"' in command
    assert '[[ "$ci_conclusion" != "success" ]]' in command


def test_ci_conformance_receipt_is_created_and_uploaded_in_the_parity_job():
    jobs = _workflow_jobs("ci.yml")
    parity = jobs["official-parity"]
    run_step = _named_step(parity, "Compare full, cached, backward, and MTP execution")
    upload_step = _named_step(parity, "Upload the offline conformance receipt")

    assert "--profile offline" in run_step["run"]
    assert "/tmp/nano-deepseek-v4-conformance-offline.json" in run_step["run"]
    assert upload_step["with"]["path"] == (
        "/tmp/nano-deepseek-v4-conformance-offline.json"
    )
    assert upload_step["with"]["if-no-files-found"] == "error"


def test_offline_conformance_receipts_use_verified_network_namespaces():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    ci_jobs = _workflow_jobs("ci.yml")
    release_jobs = _workflow_jobs("release.yml")

    assert ci_jobs["official-parity"]["runs-on"] == "ubuntu-24.04"
    assert ci_jobs["package"]["runs-on"] == "ubuntu-24.04"
    assert release_jobs["build"]["runs-on"] == "ubuntu-24.04"

    ci_offline_calls = ci.count("--profile core") + ci.count("--profile offline")
    release_offline_calls = release.count("--profile core") + release.count(
        "--profile offline"
    )
    assert ci_offline_calls == 4
    assert release_offline_calls == 3
    # In addition to every core/offline profile, isolate sdist and wheel
    # reproduce/generate/tour plus the installed notebook test and execution.
    isolated_non_conformance_calls = 8
    assert ci.count('"$GITHUB_WORKSPACE/scripts/run_no_network.py"') == (
        ci_offline_calls + isolated_non_conformance_calls
    )
    assert release.count('"$GITHUB_WORKSPACE/scripts/run_no_network.py"') == (
        release_offline_calls + isolated_non_conformance_calls
    )
    assert ci.count("--require-os-network-isolation") == ci_offline_calls
    assert release.count("--require-os-network-isolation") == release_offline_calls
    assert ci.count('"$GITHUB_WORKSPACE/tests/test_notebook.py"') == 1
    assert release.count('"$GITHUB_WORKSPACE/tests/test_notebook.py"') == 1
    assert ci.count("/bin/jupyter nbconvert") == 2
    assert release.count("/bin/jupyter nbconvert") == 2

    live = _named_step(
        release_jobs["build"], "Run the live pinned conformance profile"
    )["run"]
    assert "scripts/run_no_network.py" not in live
    assert "--require-os-network-isolation" not in live
    assert 'network.get("blocked_attempt_count") != 0' in live
    assert 'network.get("os_isolation") != "none"' in live


def test_release_receipt_and_publication_steps_have_the_required_job_dependencies():
    jobs = _workflow_jobs("release.yml")
    build = jobs["build"]
    step_names = [step.get("name") for step in build["steps"]]

    source_index = step_names.index("Verify release source and version")
    cuda_index = step_names.index("Verify and stage the CUDA release receipt")
    build_index = step_names.index("Build and validate distributions")
    smoke_index = step_names.index("Smoke-test the release wheel")
    live_index = step_names.index("Run the live pinned conformance profile")
    cuda_reverify_index = step_names.index(
        "Reverify the CUDA receipt against the release source"
    )
    identity_index = step_names.index("Record release identity")
    upload_index = step_names.index("Upload distributions")
    assert (
        source_index
        < cuda_index
        < build_index
        < smoke_index
        < live_index
        < cuda_reverify_index
        < identity_index
        < upload_index
    )

    cuda = _named_step(build, "Verify and stage the CUDA release receipt")["run"]
    cuda_reverify = _named_step(
        build,
        "Reverify the CUDA receipt against the release source",
    )["run"]
    live = _named_step(build, "Run the live pinned conformance profile")["run"]
    identity = _named_step(build, "Record release identity")["run"]
    upload = _named_step(build, "Upload distributions")["with"]
    assert "python scripts/cuda_release_gate.py verify \\" in cuda
    assert "--receipt references/cuda-release-gate.json" in cuda
    assert 'git merge-base --is-ancestor "$tested_commit" "$GITHUB_SHA"' in cuda
    assert (
        "cp -- references/cuda-release-gate.json \\\n"
        "  dist/cuda/cuda-release-gate.json"
    ) in cuda
    assert "python scripts/cuda_release_gate.py verify \\" in cuda_reverify
    assert "--receipt dist/cuda/cuda-release-gate.json" in cuda_reverify
    assert "--profile live" in live
    assert "dist/conformance/core.json" in live
    assert '"conformance_receipts": conformance_receipts' in identity
    assert '"cuda_receipts": cuda_receipts' in identity
    assert upload["path"] == "dist/"
    assert upload["if-no-files-found"] == "error"

    assert jobs["attest"]["needs"] == "build"
    assert set(jobs["publish-pypi"]["needs"]) == {"build", "attest"}
    pypi_publish = _named_step(
        jobs["publish-pypi"],
        "Publish distributions with PyPI Trusted Publishing",
    )["with"]
    assert pypi_publish["packages-dir"] == "dist/packages/"
    assert pypi_publish["skip-existing"] is True
    assert jobs["verify-pypi"]["needs"] == "publish-pypi"
    assert jobs["publish-github"]["needs"] == "verify-pypi"

    attest = _named_step(jobs["attest"], "Attest release evidence")["with"]
    subjects = attest["subject-path"]
    assert "dist/packages/*" in subjects
    assert "dist/conformance/*" in subjects
    assert "dist/cuda/*" in subjects
    assert "dist/release-manifest.json" in subjects

    github_job = jobs["publish-github"]
    github_steps = [step.get("name") for step in github_job["steps"]]
    assert github_steps.index("Revalidate release identity") < github_steps.index(
        "Create, populate, and publish the GitHub release"
    )
    publish = _named_step(
        github_job, "Create, populate, and publish the GitHub release"
    )["run"]
    assert "gh release view" in publish
    assert "releases/generate-notes" in publish
    assert '-f target_commitish="$GITHUB_SHA"' in publish
    assert '--notes-file "$notes_file"' in publish
    assert "--json assets,body,isDraft,isPrerelease,name,tagName" in publish
    assert publish.count("scripts/verify_github_release.py") == 3
    assert "--allow-partial-assets" in publish
    assert "--published" in publish
    assert "--asset-directory dist/packages" in publish
    assert "--asset dist/release-manifest.json" in publish
    assert "dist/cuda/*" in publish
    assert "--clobber" in publish
    assert "gh release edit" in publish


def test_tag_cuda_release_gate_is_cpu_only_verification():
    jobs = _workflow_jobs("release.yml")
    for name in (
        "Verify and stage the CUDA release receipt",
        "Reverify the CUDA receipt against the release source",
    ):
        command = _named_step(jobs["build"], name)["run"]
        assert "cuda_release_gate.py verify" in command
        for forbidden in (
            "cuda_release_gate.py generate",
            "pytest",
            "torch",
            "nvidia-smi",
        ):
            assert forbidden not in command


@pytest.mark.parametrize(
    "workflow_path",
    [ROOT / ".github" / "workflows" / "ci.yml", ROOT / ".github" / "workflows" / "release.yml"],
)
def test_source_distribution_is_rebuilt_and_directly_smoke_tested(workflow_path: Path):
    workflow = workflow_path.read_text(encoding="utf-8")

    assert "Verify and smoke-test the source distribution" in workflow
    assert "expected exactly one built source distribution" in workflow
    assert "NANO_SDIST_PATH" in workflow
    assert "python -m pip wheel \\" in workflow
    assert "--no-deps \\" in workflow
    assert "--no-cache-dir \\" in workflow
    assert "scripts/verify_distribution_artifacts.py" in workflow
    assert "--rebuilt-wheel" in workflow
    assert '"${NANO_SDIST_PATH}"' in workflow
    assert "direct_url.json" in workflow
    assert "installed distribution is not the selected sdist" in workflow
    assert "sdist smoke imported source checkout" in workflow
    assert "CPU sdist smoke resolved accelerator packages" in workflow
    assert 'nano-deepseek-v4" demo' in workflow
    assert 'nano-deepseek-v4" conformance' in workflow
    assert "--profile core" in workflow
    assert workflow.index("Verify and smoke-test the source distribution") < workflow.index(
        "Smoke-test the", workflow.index("Verify and smoke-test the source distribution") + 1
    )


def test_public_version_surfaces_are_atomic():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert project_version is not None
    version = project_version.group(1)
    init_text = (ROOT / "nano_deepseek_v4" / "__init__.py").read_text(
        encoding="utf-8"
    )
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert re.search(
        rf'^__version__ = "{re.escape(version)}"$', init_text, re.MULTILINE
    )
    assert f"version: {version}\n" in citation
    assert f"| Published PyPI `{version}` |" in readme
    assert f'python -m pip install "nano-deepseek-v4=={version}"' in readme
    assert f"nano-deepseek-v4 {version}" in readme


def test_unified_cli_and_legacy_entry_points_are_both_packaged():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    expected = {
        'nano-deepseek-v4 = "nano_deepseek_v4.cli:main"',
        'nano-deepseek-v4-aggregate = "nano_deepseek_v4.aggregate_comparisons:main"',
        'nano-deepseek-v4-attention-reach = "nano_deepseek_v4.attention_reach:main"',
        'nano-deepseek-v4-demo = "nano_deepseek_v4.demo:cli_main"',
        'nano-deepseek-v4-generate = "nano_deepseek_v4.generate_text:main"',
        'nano-deepseek-v4-compare = "nano_deepseek_v4.compare_bundles:main"',
        'nano-deepseek-v4-inspect = "nano_deepseek_v4.architecture:main"',
        'nano-deepseek-v4-parity = "nano_deepseek_v4.official_parity:main"',
        'nano-deepseek-v4-train = "nano_deepseek_v4.train_text:main"',
        (
            'nano-deepseek-v4-verify-flash-0731 = '
            '"nano_deepseek_v4.verify_flash_0731_receipt:main"'
        ),
    }

    assert all(entry in pyproject for entry in expected)
    assert "nano-deepseek-v4-reproduce" not in pyproject


def test_installed_sdist_and_wheel_smoke_the_dspark_vectors():
    workflows = (
        ("ci.yml", "package", "Smoke-test the built wheel in an isolated environment"),
        ("release.yml", "build", "Smoke-test the release wheel"),
    )
    for filename, job_name, wheel_step_name in workflows:
        job = _workflow_jobs(filename)[job_name]
        sdist = _named_step(job, "Verify and smoke-test the source distribution")["run"]
        wheel = _named_step(job, wheel_step_name)["run"]

        assert '"_receipts/dspark-scheduler-v1.json"' in sdist
        assert '"_receipts/dspark-semantic-v1.json"' in sdist
        assert '"nano_deepseek_v4/_receipts/dspark-scheduler-v1.json"' in wheel
        assert '"nano_deepseek_v4/_receipts/dspark-semantic-v1.json"' in wheel
        assert '"$GITHUB_WORKSPACE/tests/test_dspark_conformance.py"' in wheel
        assert '"$GITHUB_WORKSPACE/tests/test_dspark_scheduler.py"' in wheel

        for installed_smoke in (sdist, wheel):
            assert re.search(r'/bin/nano-deepseek-v4"? dspark \\', installed_smoke)
            assert re.search(
                r'/bin/nano-deepseek-v4"? dspark-scheduler \\',
                installed_smoke,
            )
            assert "--profile dspark \\" in installed_smoke

        assert wheel.index("/bin/nano-deepseek-v4 dspark \\") < wheel.index(
            '"${NANO_WHEEL_PATH}[official]"'
        )


def test_installed_base_artifacts_run_and_validate_the_unified_user_journey():
    workflows = (
        ("ci.yml", "package", "Smoke-test the built wheel in an isolated environment"),
        ("release.yml", "build", "Smoke-test the release wheel"),
    )
    for filename, job_name, wheel_step_name in workflows:
        job = _workflow_jobs(filename)[job_name]
        sdist = _named_step(job, "Verify and smoke-test the source distribution")["run"]
        wheel = _named_step(job, wheel_step_name)["run"]

        for label, installed_smoke in (("sdist", sdist), ("wheel", wheel)):
            reproduce = re.search(
                r'/bin/nano-deepseek-v4"? reproduce \\', installed_smoke
            )
            generate = re.search(
                r'/bin/nano-deepseek-v4"? generate \\', installed_smoke
            )
            dspark = re.search(
                r'/bin/nano-deepseek-v4"? dspark \\', installed_smoke
            )
            tour = re.search(
                r'/bin/nano-deepseek-v4"? tour --json', installed_smoke
            )
            assert reproduce is not None, label
            assert generate is not None, label
            assert dspark is not None, label
            assert tour is not None, label
            conformance = re.compile(
                r'/bin/nano-deepseek-v4"? conformance \\'
            ).search(installed_smoke, dspark.end())
            assert conformance is not None, label
            assert reproduce.start() < generate.start() < dspark.start() < conformance.start()
            assert 'generation.get("schema_version") != 1' in installed_smoke
            assert 'generation.get("source", {}).get("kind") != "local"' in installed_smoke
            assert 'generation.get("generated_new_tokens") != 1' in installed_smoke
            assert 'generation.get("continuation_token_ids", [])' in installed_smoke
            for field in (
                "bundle_manifest_sha256",
                "config_sha256",
                "tokenizer_sha256",
                "parameter_count",
            ):
                assert f'"{field}"' in installed_smoke

        assert "installed-sdist reproduction did not pass" in sdist
        assert "installed-sdist generation receipt is invalid" in sdist
        assert "installed-wheel generation receipt is invalid" in wheel
        assert "/bin/nano-deepseek-v4-generate --help" in wheel
        assert not re.search(
            r'/bin/nano-deepseek-v4-generate \\\n\s+--bundle', wheel
        )


def test_release_runs_live_conformance_without_weight_downloads():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "Run the live pinned conformance profile" in workflow
    assert "--profile live" in workflow
    assert 'checks["flash_0731_receipt"]' in workflow
    assert 'receipt["cached_safetensors_file_count"] != 0' in workflow
    assert 'network.get("policy") != "pinned_hub_metadata_only"' in workflow


def test_ci_and_release_preserve_conformance_receipts_as_evidence():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "Upload the offline conformance receipt" in ci
    assert "conformance-offline-${{ github.sha }}" in ci
    for name in ("core.json", "dspark.json", "offline.json", "live.json"):
        assert f'dist/conformance/{name}"' in release
    assert "/tmp/nano-deepseek-v4-release-conformance-dspark.json" in release
    assert '"conformance_receipts": conformance_receipts' in release
    assert release.count('manifest.get("conformance_receipts")') == 3
    assert '"cuda_receipts": cuda_receipts' in release
    assert release.count('manifest.get("cuda_receipts")') == 3
    assert release.count("python scripts/cuda_release_gate.py verify") == 2
    assert "dist/conformance/*" in release
    assert "dist/cuda/*" in release
    assert "Attest release evidence" in release
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "clean-wheel `core`,\n`dspark`, `offline`, and `live` receipts" in readme


def test_documented_cpu_source_installs_select_torch_before_the_project():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    source_section = readme.split(
        "For every Unreleased command in this checkout, use the same CPU-first order:",
        1,
    )[1].split("CUDA users", 1)[0]
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    installation = (ROOT / "docs" / "guides" / "installation.md").read_text(
        encoding="utf-8"
    )
    install_source_section = installation.split("## CPU — current source", 1)[1].split(
        "## CUDA", 1
    )[0]

    for document in (source_section, contributing, install_source_section):
        assert "https://download.pytorch.org/whl/cpu" in document
        assert document.index("https://download.pytorch.org/whl/cpu") < document.index(
            "python -m pip install -e"
        )

    repeated_install_guides = [
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "guides" / "train-and-generate.md",
        ROOT / "docs" / "guides" / "official-checkpoints.md",
        ROOT / "docs" / "verification" / "transformers-parity.md",
    ]
    for path in repeated_install_guides:
        text = path.read_text(encoding="utf-8")
        assert "installation" in text.lower(), path
        assert "python -m pip install -e .\n" not in text, path


def test_flash_0731_live_replay_installs_the_official_extra_first():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    replay = readme.split("Replay the bundled metadata receipt", 1)[1].split(
        "A wheel built", 1
    )[0]

    assert "live, revision-pinned Hub replay" in replay
    assert replay.index('python -m pip install -e ".[official]"') < replay.index(
        "nano-deepseek-v4 verify-flash-0731 --json"
    )


def test_packaged_receipt_inventory_and_source_parity():
    receipt_root = ROOT / "nano_deepseek_v4" / "_receipts"
    expected = {
        "DeepSeek-V4-Flash-0731-config.json",
        "DeepSeek-V4-Flash-0731-metadata.json",
        "dspark-scheduler-v1.json",
        "dspark-semantic-v1.json",
        "tiny-text-training-baseline.json",
    }

    assert {path.name for path in receipt_root.glob("*.json")} == expected
    assert not (ROOT / "references" / "DeepSeek-V4-Flash-0731-config.json").exists()
    assert not (ROOT / "references" / "DeepSeek-V4-Flash-0731-metadata.json").exists()

    receipt = json.loads(
        (receipt_root / "DeepSeek-V4-Flash-0731-metadata.json").read_text()
    )
    config_digest = hashlib.sha256(
        (receipt_root / "DeepSeek-V4-Flash-0731-config.json").read_bytes()
    ).hexdigest()
    assert config_digest == receipt["official_source_sha256"]["config.json"]
    assert (receipt_root / "tiny-text-training-baseline.json").read_bytes() == (
        ROOT / "references" / "tiny-text-training-baseline.json"
    ).read_bytes()
    dspark = json.loads((receipt_root / "dspark-semantic-v1.json").read_text())
    assert dspark["kind"] == "dspark-semantic-vectors"
    assert dspark["fixture_sha256"] == (
        "6aff9f6a25e38603a358aeddcec9e411c1f961f5b3c8736111b3508b69a2138e"
    )
    scheduler = json.loads((receipt_root / "dspark-scheduler-v1.json").read_text())
    assert scheduler["kind"] == "dspark-scheduler-vectors"
    assert scheduler["fixture_sha256"] == (
        "1a61f7ff2fa70d2c73042eed3bccb85583ce3f5fb2a7bd07f8763d55ca71093f"
    )

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'nano_deepseek_v4 = ["py.typed", "_receipts/*.json"]' in pyproject
    assert (
        'nano-deepseek-v4-verify-flash-0731 = '
        '"nano_deepseek_v4.verify_flash_0731_receipt:main"'
    ) in pyproject


def test_source_checkout_verifier_wrapper_remains_compatible():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_flash_0731_receipt.py"), "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--receipt" in completed.stdout
    assert "--config" in completed.stdout


def test_only_release_runs_live_conformance_after_wheel_smoke():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "$GITHUB_WORKSPACE/scripts/verify_flash_0731_receipt.py" not in ci
    assert "$GITHUB_WORKSPACE/scripts/verify_flash_0731_receipt.py" not in release
    assert "--profile live" not in ci
    assert "Run the live pinned conformance profile" in release
    assert "timeout 300s" in release
    assert "/bin/nano-deepseek-v4 conformance" in release
    assert "--profile live" in release
    assert 'payload["status"] != "pass"' in release
    assert release.index("Smoke-test the release wheel") < release.index(
        "Run the live pinned conformance profile"
    )
