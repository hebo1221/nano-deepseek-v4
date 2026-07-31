# Releasing

Releases are deliberate version-tag events. Pushing a release tag builds and
smoke-tests one wheel and one source distribution in an unprivileged job.
Separate jobs verify and attest those exact files, upload them to PyPI through
OpenID Connect, and finally create the public GitHub release with the same
files. No long-lived PyPI token is stored in GitHub.

## One-time repository and PyPI setup

Enable
[immutable releases](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes)
for the repository before publishing. The release workflow uploads assets while
the release is a draft and publishes it only after every upload succeeds, which
lets GitHub lock the tag and assets on publication.

Add an active tag ruleset for `refs/tags/v*` that prevents updates and deletion
while still allowing initial creation. This protects the tag during the PyPI
environment-approval window, before release immutability takes effect.

Configure a PyPI pending trusted publisher with these exact values:

- PyPI project: `nano-deepseek-v4`
- GitHub owner: `hebo1221`
- repository: `nano-deepseek-v4`
- workflow: `release.yml`
- environment: `pypi`

Create the matching `pypi` environment in GitHub *before* pushing a tag,
restrict deployment branches and tags to `v*`, and add a required reviewer.
Otherwise GitHub can create an unprotected environment automatically when the
workflow first reaches the publishing job.

Protect `main` and require the `CI / Required` check. Keep CodeQL and dependency
review enabled as separate security gates.

## Release checklist

1. Update the version in `pyproject.toml`, `nano_deepseek_v4/__init__.py`, and
   `CITATION.cff`.
2. Move the relevant `CHANGELOG.md` entries from `Unreleased` to a dated
   version section.
3. Run the local gates:

   ```bash
   ruff check .
   yamllint .github .yamllint.yml
   mypy nano_deepseek_v4
   pytest --cov=nano_deepseek_v4 --cov-report=term-missing --cov-fail-under=75
   python -m build
   python -m twine check --strict dist/*
   ```

4. Merge the release pull request only after CI, CodeQL, and dependency review
   pass.
5. Create and push tag `vX.Y.Z` from the merge commit on `main`. The workflow
   rejects tags whose commit is not contained in `main` or whose version differs
   from `pyproject.toml`.
6. Approve the protected `pypi` environment deployment after inspecting the
   completed build and provenance jobs.
7. Wait for the `Publish release` workflow to finish. Verify that the GitHub
   release and PyPI contain the same wheel and source distribution, then compare
   them with the attached `SHA256SUMS` and `release-manifest.json`.
8. Install the exact version in a clean environment and run:

   ```bash
   python -m pip install "nano-deepseek-v4==X.Y.Z"
   nano-deepseek-v4-demo
   ```

## Failure recovery

- If no file reached PyPI and the failure was transient, rerun the failed jobs.
  If workflow or source changes are required, bump the version and create a new
  tag; a rerun still uses the original tagged workflow.
- If PyPI succeeded but GitHub publishing failed, rerun only the failed GitHub
  job. Do not restart the full workflow.
- If a PyPI upload is partial or ambiguous, inspect the published filenames and
  hashes before doing anything else. Never rebuild different bytes under the
  same version; complete only an identical missing upload, or yank and bump the
  version.
- If GitHub asset upload fails, inspect and remove any leftover draft release
  before retrying.
