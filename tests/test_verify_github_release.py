from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_github_release import (
    GitHubReleaseVerificationError,
    collect_asset_inventory,
    main,
    verify_github_release,
)

TAG = "v0.3.0"
NOTES = "## What's Changed\n\n- Real release notes.\n"
EXPECTED_ASSETS = {"package.whl": 11, "release-manifest.json": 22}


def _release(*, assets: dict[str, int] | None = None) -> dict[str, object]:
    return {
        "tagName": TAG,
        "name": TAG,
        "isDraft": True,
        "isPrerelease": False,
        "body": NOTES,
        "assets": [
            {"name": name, "size": size}
            for name, size in (assets or {}).items()
        ],
    }


def test_github_release_accepts_expected_partial_draft_before_upload():
    actual = verify_github_release(
        _release(assets={"package.whl": 999}),
        tag=TAG,
        expected_notes=NOTES,
        expected_assets=EXPECTED_ASSETS,
        require_complete_assets=False,
    )

    assert actual == {"package.whl": 999}


def test_github_release_accepts_exact_complete_draft_after_upload():
    actual = verify_github_release(
        _release(assets=EXPECTED_ASSETS),
        tag=TAG,
        expected_notes=NOTES.rstrip("\n"),
        expected_assets=EXPECTED_ASSETS,
        require_complete_assets=True,
    )

    assert actual == EXPECTED_ASSETS


def test_github_release_accepts_exact_public_release_after_publish():
    release = _release(assets=EXPECTED_ASSETS)
    release["isDraft"] = False

    actual = verify_github_release(
        release,
        tag=TAG,
        expected_notes=NOTES,
        expected_assets=EXPECTED_ASSETS,
        require_complete_assets=True,
        expected_draft=False,
    )

    assert actual == EXPECTED_ASSETS


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tagName", "v9.9.9", "identity drift"),
        ("name", "Custom title", "identity drift"),
        ("isDraft", False, "identity drift"),
        ("isPrerelease", True, "identity drift"),
        ("body", "stale notes", "notes differ"),
    ],
)
def test_github_release_rejects_identity_or_notes_drift(
    field: str,
    value: object,
    message: str,
):
    release = _release()
    release[field] = value

    with pytest.raises(GitHubReleaseVerificationError, match=message):
        verify_github_release(
            release,
            tag=TAG,
            expected_notes=NOTES,
            expected_assets=EXPECTED_ASSETS,
            require_complete_assets=False,
        )


def test_github_release_rejects_unexpected_draft_asset():
    with pytest.raises(GitHubReleaseVerificationError, match="unexpected assets"):
        verify_github_release(
            _release(assets={"old-debug-log.txt": 3}),
            tag=TAG,
            expected_notes=NOTES,
            expected_assets=EXPECTED_ASSETS,
            require_complete_assets=False,
        )


@pytest.mark.parametrize(
    "assets",
    [
        {"package.whl": 11},
        {"package.whl": 12, "release-manifest.json": 22},
    ],
)
def test_github_release_rejects_incomplete_or_wrong_size_final_assets(
    assets: dict[str, int],
):
    with pytest.raises(GitHubReleaseVerificationError, match="inventory mismatch"):
        verify_github_release(
            _release(assets=assets),
            tag=TAG,
            expected_notes=NOTES,
            expected_assets=EXPECTED_ASSETS,
            require_complete_assets=True,
        )


def test_asset_inventory_rejects_duplicate_basenames(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "same.json").write_text("left", encoding="utf-8")
    (right / "same.json").write_text("right", encoding="utf-8")

    with pytest.raises(GitHubReleaseVerificationError, match="duplicated"):
        collect_asset_inventory(
            assets=[],
            asset_directories=[left, right],
        )


def test_github_release_verifier_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    package_dir = tmp_path / "packages"
    package_dir.mkdir()
    package = package_dir / "package.whl"
    package.write_bytes(b"wheel")
    manifest = tmp_path / "release-manifest.json"
    manifest.write_bytes(b"manifest")
    notes = tmp_path / "notes.md"
    notes.write_text(NOTES, encoding="utf-8")
    release = tmp_path / "release.json"
    release.write_text(
        json.dumps(
            _release(
                assets={
                    package.name: package.stat().st_size,
                    manifest.name: manifest.stat().st_size,
                }
            )
        ),
        encoding="utf-8",
    )

    assert main(
        [
            "--release-json",
            str(release),
            "--tag",
            TAG,
            "--notes-file",
            str(notes),
            "--asset-directory",
            str(package_dir),
            "--asset",
            str(manifest),
        ]
    ) == 0
    assert "PASS (complete, 2 assets)" in capsys.readouterr().out
