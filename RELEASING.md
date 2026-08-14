# Releasing

Releases are deliberate version-tag events. Pushing a release tag builds and
smoke-tests one wheel and one source distribution in an unprivileged job. The
source distribution is inventory-checked, rebuilt to a wheel and compared with
the direct wheel, then installed from the exact archive in its own clean CPU
environment before the no-download demo runs.
Separate jobs verify and attest those exact files, upload them to PyPI through
OpenID Connect, verify the public PyPI filename and SHA-256 inventory, and
finally create the public GitHub release with the same files. No long-lived
PyPI token is stored in GitHub.

The build also captures `core`, `dspark`, `offline`, and `live` conformance
receipts from the clean wheel environment. Their hashes are bound into
`release-manifest.json`; the receipts, manifest, checksums, wheel, and source
distribution are attested together and published as GitHub release evidence.
The receipts are not Python distributions and are not uploaded to PyPI.
For releases that touch CUDA-sensitive code, the same manifest also binds the
checked-in, source-inventory-matched manual CUDA receipt described below.

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
   `CITATION.cff`. In the same change, update the README's published-version
   table, pinned install command, and expected demo version. Remove the
   transitional “Unreleased”, “until the next release”, and “current source”
   capability qualifiers from the README, installation, training/generation,
   official-checkpoint, and conformance guides. Replace editable conformance
   installs with version-pinned PyPI installs. Keep
   `scripts/verify_release_source.py`'s transition-fragment list synchronized
   with all five documents so the local preflight and tag job reject every
   stale source-only capability claim.
2. Move the relevant `CHANGELOG.md` entries from `Unreleased` to a dated
   version section. Add that version's comparison link from the previous tag and
   advance the `[Unreleased]` comparison link to the new tag.
3. Run the local gates:

   ```bash
   python scripts/verify_release_source.py --tag vX.Y.Z
   ruff check .
   yamllint .github .yamllint.yml
   mypy nano_deepseek_v4 scripts tests
   pytest --cov=nano_deepseek_v4 --cov-report=term-missing --cov-fail-under=75
   nano-deepseek-v4 attention-reach
   nano-deepseek-v4 tour --json
   nano-deepseek-v4 dspark --json --output /tmp/dspark.json
   nano-deepseek-v4 dspark-scheduler --json --output /tmp/dspark-scheduler.json
   nano-deepseek-v4 parity
   nano-deepseek-v4 verify-flash-0731 --json
   nano-deepseek-v4 conformance --profile core --json --output /tmp/conformance-core.json
   nano-deepseek-v4 conformance --profile dspark --json --output /tmp/conformance-dspark.json
   nano-deepseek-v4 conformance --profile offline --json --output /tmp/conformance-offline.json
   nano-deepseek-v4 conformance --profile live --json --output /tmp/conformance-live.json
   release_check_dir="$(mktemp -d /tmp/nano-deepseek-v4-release-check.XXXXXX)"
   mkdir -p "$release_check_dir/packages" "$release_check_dir/rebuilt"
   python -m build --outdir "$release_check_dir/packages"
   python -m twine check --strict "$release_check_dir"/packages/*
   python -m pip wheel \
     --no-deps \
     --no-cache-dir \
     --wheel-dir "$release_check_dir/rebuilt" \
     "$release_check_dir"/packages/*.tar.gz
   python scripts/verify_distribution_artifacts.py \
     --wheel "$release_check_dir"/packages/*.whl \
     --sdist "$release_check_dir"/packages/*.tar.gz \
     --rebuilt-wheel "$release_check_dir"/rebuilt/*.whl
   ```

   Run the release-source verifier only after steps 1 and 2 are complete. It is
   intentionally stricter than ordinary development CI: an active, non-empty
   `Unreleased` section or any source-only capability qualifier makes it fail.
   The tag workflow runs the same executable after independently checking that
   the tagged commit is contained in `origin/main`.

   The direct local `core`, `dspark`, and `offline` commands exercise the in-process
   standard-library socket guard, which is intentionally not described as a
   sandbox. On a Linux host with passwordless `sudo`, `unshare`, `ip`, and
   `setpriv`, the release-equivalent no-network form is:

   ```bash
   python scripts/run_no_network.py \
     nano-deepseek-v4 conformance \
     --profile offline \
     --require-os-network-isolation \
     --json \
     --output /tmp/conformance-offline-netns.json
   ```

   The wrapper fails closed unless the network namespace changed, only an active
   `lo` remains, and effective capabilities are zero. Bringing up loopback lets
   a Jupyter kernel communicate only inside the isolated namespace; there is no
   egress interface. The hosted CI and tag workflow always use this form for
   every `core` and `offline` release receipt and the notebook smoke. They also
   run both standalone DSpark vector checks and the `dspark` profile from the
   installed artifact; `live` remains outside the namespace by design.

   Hosted CI and the tag workflow are CPU-only. Every release must carry a
   current manual CUDA receipt from a compatible host. An existing receipt can
   be reused only while its tested commit remains an ancestor and every covered
   source byte is unchanged; otherwise generate a new one.

   First finish and commit every covered source change. Starting from that clean
   commit, run:

   ```bash
   python scripts/cuda_release_gate.py generate \
     --output references/cuda-release-gate.json
   python scripts/cuda_release_gate.py verify \
     --receipt references/cuda-release-gate.json
   ```

   The generator discovers the exact `gpu`-marked accelerator node IDs under the
   checked-in `pyproject.toml`, with ambient pytest options/plugins and
   `conftest.py` disabled. It requires every selected node to pass with zero
   skips, captures Python, PyTorch, CUDA, driver, and device provenance, and
   binds the result to a sorted byte inventory of the package Python sources,
   `py.typed`, `pyproject.toml`, the accelerator test, and the gate script
   itself. Every covered working file is compared directly with its blob in the
   recorded commit, so Git index flags cannot hide a changed test input. The
   `--allow-dirty` option is only for a temporary local dogfood receipt; such a
   receipt records `worktree_clean=false` and the verifier rejects it as release
   evidence.

   Commit only the generated `references/cuda-release-gate.json` after the clean
   tested source commit, then tag that descendant commit. The tag workflow does
   not rerun CUDA. It uses the standard-library-only verifier before the build
   and again after the wheel smoke/live checks to require exact current inventory
   bytes, requires the receipt's tested commit to remain an ancestor of the tag,
   copies the receipt into `dist/cuda`, hashes it into the release manifest, and
   publishes and attests those exact JSON bytes. This is artifact provenance,
   not a claim that the hosted runner had a GPU or that GitHub remotely attested
   the physical test machine.

   If a squash or rebase removes the tested source commit from `main`, regenerate
   the receipt from the final clean `main` commit. Any later covered-source edit
   also invalidates the receipt and requires a fresh GPU run.

4. Merge the release pull request only after CI, CodeQL, and dependency review
   pass.
5. Create and push tag `vX.Y.Z` from the merge commit on `main`. The workflow
   rejects tags whose commit is not contained in `main` or whose version differs
   from `pyproject.toml`. Before building, it also queries GitHub Actions and
   requires a completed, successful `ci.yml` push run on `main` for the exact
   tag commit SHA.
6. Approve the protected `pypi` environment deployment after inspecting the
   completed build and provenance jobs.
7. Wait for the `Publish release` workflow to finish. Its separate `Verify PyPI
   publication` job waits for the version JSON API and requires its complete
   filename and SHA-256 inventory to match `release-manifest.json` before the
   GitHub release can be created. Independently spot-check both public pages.
8. Install the exact version in a clean environment and run:

   ```bash
   python -m pip install \
     --index-url https://download.pytorch.org/whl/cpu \
     "torch>=2.4"
   python -m pip install "nano-deepseek-v4[official,parity]==X.Y.Z"
   nano-deepseek-v4 --version
   nano-deepseek-v4 demo
   nano-deepseek-v4 tour
   python -m nano_deepseek_v4 --help
   nano-deepseek-v4 inspect --preset flash
   nano-deepseek-v4 inspect --preset flash-0731 --json
   nano-deepseek-v4 verify-flash-0731 --json
   nano-deepseek-v4 attention-reach
   nano-deepseek-v4 dspark --json
   nano-deepseek-v4 dspark-scheduler --json
   nano-deepseek-v4 conformance --profile core
   nano-deepseek-v4 conformance --profile dspark
   nano-deepseek-v4 conformance --profile offline
   nano-deepseek-v4 conformance --profile live
   nano_release_smoke_root="$(mktemp -d)"
   nano-deepseek-v4 reproduce \
     --save-directory "${nano_release_smoke_root}/bundle" \
     --output "${nano_release_smoke_root}/reproduction.json" \
     --json
   python - "${nano_release_smoke_root}/reproduction.json" <<'PY'
   import json
   import sys
   from pathlib import Path

   receipt = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
   acceptance = receipt.get("acceptance", {})
   artifact = receipt.get("artifact", {})
   if receipt.get("status") != "pass" or acceptance.get("passed") is not True:
       raise SystemExit("fixed reproduction did not pass")
   if artifact.get("published") is not True:
       raise SystemExit("fixed reproduction bundle was not published")
   PY
   nano-deepseek-v4 inspect --bundle "${nano_release_smoke_root}/bundle"
   nano-deepseek-v4 generate \
     --bundle "${nano_release_smoke_root}/bundle" \
     --prompt "DeepSeek" \
     --max-new-tokens 1 \
     --device cpu \
     --json
   nano-deepseek-v4 parity
   ```

9. Follow the [release receipt procedure](docs/verification/conformance.md#release-receipts)
   for the new tag. Verify `release-manifest.json` before using its receipt
   digests, then verify the attestation for each conformance profile and the
   source-bound CUDA receipt retained as release evidence.

## Failure recovery

- If no file reached PyPI and the failure was transient, rerun the failed jobs.
  If workflow or source changes are required, bump the version and create a new
  tag; a rerun still uses the original tagged workflow.
- A transient Hub availability failure in the pre-publication Flash-0731
  receipt replay may be rerun. Hash, schema, or semantic receipt drift requires
  a corrected commit and a new version and tag; do not publish from the stale
  tag.
- If the upload succeeded but only `Verify PyPI publication` failed, inspect the
  version JSON and rerun that failed job after PyPI finishes propagating. The
  upload job does not need to run again. A full workflow rerun is also safe for
  the same immutable tag: the publisher skips existing filenames, then the
  verifier still requires the complete public filename and SHA-256 inventory to
  equal `release-manifest.json`.
- If PyPI succeeded but GitHub publishing failed, rerun only the failed GitHub
  job when possible. A full workflow rerun uses the same skip-and-verify PyPI
  path; it never treats a skipped filename as proof that its bytes match.
- If a PyPI upload is partial or ambiguous, inspect the published filenames and
  hashes before doing anything else. Never rebuild different bytes under the
  same version; complete only an identical missing upload, or yank and bump the
  version.
- If GitHub asset upload fails, rerun only `publish-github`. After revalidating
  the tag, it reuses a draft only when the title, generated notes, prerelease
  state, and every existing asset name are exactly expected. It replaces
  partial expected assets, requires the final filenames and byte sizes to match
  the local evidence, and rechecks the public state after publishing. An
  unexpected draft asset or edited metadata fails closed; inspect it instead of
  deleting it blindly. Already published releases are never modified.
