from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_flash_inference.ipynb"


def _load_notebook() -> dict[str, Any]:
    payload = json.loads(_NOTEBOOK.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _cell_source(cell: dict[str, Any]) -> str:
    source = cell["source"]
    if isinstance(source, str):
        return source
    assert isinstance(source, list)
    assert all(isinstance(line, str) for line in source)
    return "".join(source)


def _code_cells(notebook: dict[str, Any]) -> list[dict[str, Any]]:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return [cell for cell in cells if cell["cell_type"] == "code"]


def test_flash_tutorial_is_clean_compilable_and_current():
    notebook = _load_notebook()
    cells = _code_cells(notebook)

    assert notebook["nbformat"] == 4
    assert cells
    for index, cell in enumerate(cells):
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile(_cell_source(cell), f"{_NOTEBOOK.name}:cell-{index}", "exec")

    source = "\n".join(_cell_source(cell) for cell in notebook["cells"])
    assert "DeepSeekV4Config.flash_0731()" in source
    assert "run_tiny_text_training" in source
    assert "run_text_generation" in source
    assert "run_conformance(profile='core')" in source
    assert "full_flash_coverage.recognized_keys" in source
    assert "full_flash_coverage.scale_keys" in source
    assert "full_flash_coverage.unrecognized_keys" in source
    assert "coverage.total_keys" not in source
    assert "pattern_counts" not in source
    assert "U8" not in source
    assert "signed I8" in source
    assert "DOWNLOAD_FULL_FLASH = False" in source
    assert "RUN_FULL_FLASH_PREFLIGHT = False" in source
    assert "MATERIALIZE_FULL_FLASH = False" in source


def test_flash_tutorial_default_sequence_runs_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    notebook = _load_notebook()
    network_attempts: list[str] = []

    def block_network(*_args: object, **_kwargs: object) -> None:
        network_attempts.append("attempted")
        raise OSError("network access is forbidden in the notebook smoke test")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(socket, "create_connection", block_network)
    monkeypatch.setattr(socket, "getaddrinfo", block_network)
    monkeypatch.setattr(socket.socket, "connect", block_network)
    monkeypatch.setattr(socket.socket, "connect_ex", block_network)
    monkeypatch.setattr(socket.socket, "sendto", block_network)

    namespace: dict[str, Any] = {"__name__": "__notebook__"}
    workspace: Any = None
    try:
        for index, cell in enumerate(_code_cells(notebook)):
            code = compile(
                _cell_source(cell),
                f"{_NOTEBOOK.name}:cell-{index}",
                "exec",
            )
            exec(code, namespace)
        workspace = namespace["_tutorial_workspace"]

        bundle_dir = namespace["bundle_dir"]
        training_result = namespace["training_result"]
        generation_result = namespace["generation_result"]
        conformance_report = namespace["conformance_report"]
        flash_0731_architecture = namespace["flash_0731_architecture"]
        flash_0731_metadata = namespace["flash_0731_metadata"]

        assert bundle_dir.is_dir()
        assert (bundle_dir / "nano_deepseek_v4.json").is_file()
        assert training_result.steps == 1
        assert training_result.bundle_format_version == 2
        assert training_result.checkpoint_round_trip_match is True
        assert training_result.tokenizer_round_trip_match is True
        assert generation_result.generated_new_tokens == 1
        assert generation_result.bundle_manifest_sha256 == training_result.bundle_manifest_sha256
        assert conformance_report.passed
        assert conformance_report.complete
        assert conformance_report.network["policy"] == "forbidden"
        assert conformance_report.network["attempted"] is False
        assert flash_0731_architecture.auxiliary_kind == "dspark"
        assert flash_0731_architecture.dspark_stage_count == 3
        assert flash_0731_architecture.runtime_load_supported is False
        assert flash_0731_metadata["dspark_namespace"]["dtype_counts"]["I8"] == 2304
        assert namespace["DOWNLOAD_FULL_FLASH"] is False
        assert namespace["RUN_FULL_FLASH_PREFLIGHT"] is False
        assert namespace["MATERIALIZE_FULL_FLASH"] is False
        assert namespace["full_flash_preflight"] is None
        assert namespace["full_flash_coverage"] is None
        assert namespace["full_flash_load_report"] is None
        assert network_attempts == []
    finally:
        if workspace is None:
            workspace = namespace.get("_tutorial_workspace")
        if workspace is not None:
            workspace.cleanup()
