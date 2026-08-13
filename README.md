# nano-deepseek-v4

[![CI](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/ci.yml/badge.svg)](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/ci.yml)
[![CodeQL](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/codeql.yml/badge.svg)](https://github.com/hebo1221/nano-deepseek-v4/actions/workflows/codeql.yml)
[![PyPI](https://img.shields.io/pypi/v/nano-deepseek-v4.svg)](https://pypi.org/project/nano-deepseek-v4/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://github.com/hebo1221/nano-deepseek-v4/blob/main/LICENSE)

A CPU-runnable conformance and reference lab for the **DeepSeek-V4**
architecture. The native PyTorch equations remain readable in
`nano_deepseek_v4/modeling.py`, while deterministic receipts and allocation-free
inspection make it possible to check specific claims without materializing an
official checkpoint.

The native model follows the
[DeepSeek-V4 technical report](https://arxiv.org/abs/2606.19348) and includes
sliding attention, Compressed Sparse Attention
(CSA), Heavily Compressed Attention (HCA), official compressed-branch YaRN,
Manifold-Constrained Hyper-Connections (mHC), hash and learned MoE routing, the
Lightning Indexer, and Multi-Token Prediction (MTP). It also provides native
checksummed bundles, official checkpoint inspection, and small reproducible
training and verification tools. For DeepSeek-V4-Flash-0731, the repository
separates a pinned metadata inspector, a tiny CPU DSpark equation checker, and a
CPU prefix-scheduler checker from the still-unsupported official-checkpoint
speculative runtime.

| Evidence surface | What it establishes | What it does not establish |
| --- | --- | --- |
| Conformance profiles | One fixed run records local execution, structure, reachability, and selected independent-reference results | A general model or systems benchmark |
| CPU demo | Full and cached execution agree on the complete tiny stack | Text quality |
| Fixed CPU reproduction | The installed package learns the fixed tiny corpus and persists a verified native bundle | Cross-runtime byte identity, language quality, or official training parity |
| Pinned Transformers parity | Tiny eager full/cache/backward/MTP equations agree | Frontier kernels or distributed parity |
| DSpark semantic vectors | A separate native path and dense oracle reproduce fixed 3-stage, 5-token draft intermediates and proposals | Official weights, acceptance, scheduling, speed, or quality |
| DSpark scheduler vectors | Algorithm 1 and Section 5.2 CPU arithmetic and causal traces agree with an exhaustive oracle on supplied calibrated probabilities and synthetic SPS tables | Input temporal provenance or slot alignment, calibration, measured hardware, acceptance, losslessness, speed, serving, quality, or official-weight runtime |
| Pinned 0731 DSpark preflight | Config, index, and three selected shard headers agree with the 3-stage DSpark schema | Weight payload integrity or speculative execution |
| Attention reachability check | The native hybrid graph carries influence beyond the matched local boundary | Learned long-context retrieval |
| Tiny Shakespeare control | The hybrid stack learns real text and round-trips through a bundle | Hybrid quality superiority or seed-level stability |

The defaults are deliberately tiny. The library can inspect the official
284B Flash and 1.6T Pro layouts, but materializing those models still requires
correspondingly large memory. This is a reference and conformance surface, not
a production training or serving framework.

## Sixty-second CPU demo

Choose the install that matches the commands you intend to run:

| Install | Contains |
| --- | --- |
| Published PyPI `0.2.0` | Core model, cache, checkpoint tools, and `nano-deepseek-v4-demo` |
| Current source checkout | The Unreleased unified CLI, training, generation, parity, inspection, and research tools documented below |

For the published CPU release, install the CPU PyTorch wheel first so a clean
Linux environment does not resolve PyTorch's much larger CUDA dependency set:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.4"
python -m pip install "nano-deepseek-v4==0.2.0"
```

For every Unreleased command in this checkout, use the same CPU-first order:

```bash
git clone https://github.com/hebo1221/nano-deepseek-v4.git
cd nano-deepseek-v4
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch>=2.4"
python -m pip install -e .
```

CUDA users should install the PyTorch build appropriate for their system before
installing this package. Python 3.10 or newer and PyTorch 2.4 or newer are
required. The full CPU, CUDA, extras, and environment-check procedure is in the
[installation guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/guides/installation.md).

Run the no-download architecture and cache check:

```bash
nano-deepseek-v4-demo
# Equivalent: python -m nano_deepseek_v4.demo
```

The current source also provides one discoverable command surface:

```bash
nano-deepseek-v4 --help
nano-deepseek-v4 demo
# Equivalent: python -m nano_deepseek_v4 demo
```

The separate `nano-deepseek-v4-*` commands remain supported for compatibility
with the published release and existing scripts.

Expected structure:

```text
nano-deepseek-v4 0.2.0
parameters: 1,023,364
attention: sliding -> sliding -> compressed_sparse -> heavily_compressed
mlp: hash_moe -> hash_moe -> hash_moe -> moe
logits: (1, 8, 512)
cache tokens: 5 -> 8
cached/full match: True
router layers: 4
```

The weights and token IDs are random. This proves that the architecture and
chunked cache path execute consistently; it is not a text-quality demo.

## Run a conformance profile

The current source combines the local execution checks into one fixed,
no-download CPU profile:

```bash
nano-deepseek-v4 conformance --profile core
nano-deepseek-v4 conformance --profile dspark
nano-deepseek-v4 conformance \
  --profile core \
  --json \
  --output conformance-core.json
```

`core` runs the native tiny full/cache check, allocation-free Flash-0731
architecture consistency, and attention reachability. `dspark` branches from
`core` and adds separate fixed CPU draft-equation and prefix-scheduler vectors.
`offline` adds the exact
Transformers 5.15.0 parity oracle to `core`; `live` adds a revision-pinned Hub
metadata/header replay and is the only profile that permits network access. No
profile loads official weights. The live replay requests metadata and selected
header ranges and caches no weight files; it neither verifies payload integrity
nor guarantees that zero payload bytes traverse the network.

Run only the DSpark equation check when you want the smallest portable receipt:

```bash
nano-deepseek-v4 dspark --json --output dspark.json
```

It compares an independently written dense FP32 oracle, a separate native
implementation, and packaged golden inputs, weights, intermediates, and exact
draft IDs. Six deliberately broken variants guard target-layer order,
within-block non-causality, Markov recurrence, and confidence semantics. See the
[DSpark vector contract](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/verification/dspark.md).

Run only the scheduler arithmetic and causal-trace check with:

```bash
nano-deepseek-v4 dspark-scheduler \
  --json \
  --output dspark-scheduler.json
```

It covers the paper's Algorithm 1 and Section 5.2 using supplied calibrated
probabilities and synthetic SPS tables, including strict causal stopping and
the caller-supplied lagged capacity rule. It cannot verify that the history was
captured exactly two steps earlier or kept aligned to the same request slots.
It is not an STS calibration, measured hardware, rejection-sampling,
losslessness, speed, serving, quality, or official-weight runtime result. See the
[DSpark scheduler contract](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/verification/dspark-scheduler.md).

The JSON receipt records dependency versions, implementation and nested-receipt
digests, every skipped check, the network policy and enforcement scope, and the
claim boundary. Local no-network enforcement is a best-effort Python
standard-library guard; CI and release `core`/`offline` receipts additionally
require a capability-dropped Linux network namespace. See
the [conformance profile guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/verification/conformance.md)
for installs, the fixed check matrix, exit codes, and limits.

Tagged releases that include the conformance command attach clean-wheel `core`,
`dspark`, `offline`, and `live` receipts to the GitHub release. Their digests are recorded
in the release manifest and covered by the same provenance attestation as the
distributions. The
[release-receipt procedure](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/verification/conformance.md)
shows how to bind a downloaded receipt back to that manifest and attestation.

## Quickstart — tiny model on CPU

```python
import torch
from nano_deepseek_v4 import DeepSeekV4Config, DeepSeekV4ForCausalLM

config = DeepSeekV4Config()
model = DeepSeekV4ForCausalLM(config)
ids = torch.randint(0, config.vocab_size, (1, 16))
out = model(ids, labels=ids)
print(out.logits.shape, out.loss.item())
```

## One-command learning smoke test

This path requires the **current source checkout** until the next release. It
runs the wheel-bundled, fixed 20-step CPU contract, verifies the resulting
tokenizer-bound native bundle, and retains it only after every portable
acceptance check passes:

```bash
nano-deepseek-v4 reproduce \
  --save-directory ./checkpoints/tiny-text
```

`--save-directory` must be absent or an empty, non-symlink directory. If
`--output` is also used, keep the receipt outside the bundle directory tree;
neither path may contain the other.

Expected output shape:

```text
nano-deepseek-v4 reproduce (tiny-text-cpu-v1)
overall: PASS
loss reduction: ... > 2.0000
bundle verification: PASS
historical observations: ...
bundle: checkpoints/tiny-text
scope: fixed tiny CPU learning and native bundle persistence; not model quality or official-weight parity
```

The hard result is deliberately portable: it checks the fixed corpus, recipe,
loss reduction, configuration identity, and complete checksum-verified v2 bundle
round trip. Exact losses, generated bytes, and bundle digest from a matching
historical implementation are reported separately as informational
observations. They may differ across runtimes without turning an otherwise
valid run into a failure.

Without `--save-directory`, the command verifies a temporary bundle and removes
it. Add `--json --output reports/tiny-text-reproduction.json` to retain the
complete machine receipt. Use `nano-deepseek-v4 train` instead when you want to
change the corpus, step count, configuration, device, or sampling parameters.

Generate from the exact bundle without guessing token IDs:

```bash
nano-deepseek-v4 generate \
  --bundle ./checkpoints/tiny-text \
  --prompt "DeepSeek V4 " \
  --max-new-tokens 40 \
  --device cpu
```

Add `--temperature 0.8 --top-p 0.9 --seed 42` for reproducible sampling.
`--json` records the exact continuation token IDs and bundle, config, and
tokenizer digests. The built-in corpus is intentionally tiny and byte-level;
this is a learning and integration check, not a language-quality benchmark.
See the full
[training and generation guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/guides/train-and-generate.md).

## Interactive CPU-first tutorial

The checked-in
[tutorial notebook](https://github.com/hebo1221/nano-deepseek-v4/blob/main/notebooks/01_flash_inference.ipynb)
starts with the same no-download path: tiny forward execution, one-step
training, checksummed bundle inspection, generation, and the `core`
conformance profile. Official Flash and Flash-0731 inspection comes afterward;
the full 159.6 GB checkpoint path is an explicit opt-in appendix.

From a source checkout:

```bash
python -m pip install -e ".[notebook]"
jupyter lab notebooks/01_flash_inference.ipynb
```

The linked notebook tracks `main`, so it must be paired with the current source
checkout. Notebook files are included in source distributions but not installed
by wheels. For a wheel, use the notebook from the matching release tag instead
of mixing an older package with the mutable `main` copy:

```bash
NANO_VERSION="$(python -c 'from importlib.metadata import version; print(version("nano-deepseek-v4"))')"
python -m pip install "nano-deepseek-v4[notebook]==${NANO_VERSION}"
curl --fail --location --output 01_flash_inference.ipynb \
  "https://raw.githubusercontent.com/hebo1221/nano-deepseek-v4/v${NANO_VERSION}/notebooks/01_flash_inference.ipynb"
jupyter lab 01_flash_inference.ipynb
```

The `notebook` extra supplies the Jupyter runtime; it does not install the
repository notebook into `site-packages`.

## Train an 8.50M model on real text

The complete Tiny Shakespeare baseline, matched all-sliding control, paired
evaluation, exact digests, and null claim boundary moved to
the [Tiny Shakespeare study](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/research/tiny-shakespeare.md).
In the checked-in run, the descriptive loss interval crossed zero. The result
does not support a quality comparison or characterize training-seed variation.

## Check attention-path reachability

```bash
nano-deepseek-v4 attention-reach
```

The fixed check confirms that a three-layer window-32 local control reaches
source lag 93 and no farther, while the hybrid and full-window controls respond
beyond that boundary. It is structural conformance on random tiny models, not a
learned retrieval benchmark. Protocol, metrics, and limitations are in the
[attention reachability guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/verification/attention-reach.md).

## Verify the equations against independent references

```bash
python -m pip install -e ".[parity]"
nano-deepseek-v4 parity
```

The pinned verifier compares full forward, irregular cached decoding, backward
gradients, official-dimension YaRN, and MTP equations against independent
references. Expected result: `parity: PASS`. See the
[Transformers parity guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/verification/transformers-parity.md).

## Save and reload a native model bundle

Native v1 model-only and v2 tokenizer-bound bundle formats, atomic publication,
checksum verification, and generation-readiness rules are documented in the
[native bundle guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/guides/native-bundles.md).

## Inspecting official Flash checkpoints

Use `nano-deepseek-v4 inspect --preset flash` for allocation-free architecture
counts. The current 0731 checkpoint can be inspected without downloading its
full 166.9 GB checkpoint payload:

```bash
python -m pip install -e ".[official]"
nano-deepseek-v4 inspect \
  --hf-repo deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --namespace mtp \
  --json
```

Replay the bundled metadata receipt through the installed command. This is a
live, revision-pinned Hub replay, so the `official` extra is required even
though the canonical expected receipt is bundled in the wheel:

```bash
python -m pip install -e ".[official]"
nano-deepseek-v4 verify-flash-0731 --json
```

A wheel built from the current Unreleased source contains the command, verifier
code, and canonical
[receipt](https://github.com/hebo1221/nano-deepseek-v4/blob/main/nano_deepseek_v4/_receipts/DeepSeek-V4-Flash-0731-metadata.json)
in the same artifact. The verifier pins and hashes the four official
metadata/source files, checks the three selected shard headers, and requires
zero cached `.safetensors` files when its dedicated temporary cache is inspected
at completion. It never imports the downloaded inference source and still does
not verify checkpoint payload integrity or claim that no payload bytes crossed
the network. `python scripts/verify_flash_0731_receipt.py --json` remains a thin
source-checkout compatibility wrapper.

The pinned result recognizes three DSpark stages and 4,705 tensors in shards
46–48. It reports `runtime_load_supported=false`: this is a config/index/header
preflight, not a speculative-decoding runtime. Revision-pinned Hub inspection
and complete local snapshot verification are covered in the
[official checkpoint guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/guides/official-checkpoints.md).

## Persisting an inference cache

The cache format is checksum-bound to a model configuration and caller-supplied
model revision. Its API, validation contract, and trust boundary are in the
[cache persistence guide](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/guides/cache-persistence.md).

## Documentation

See the [documentation index](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/README.md)
for guides, verification protocols, research receipts, and source navigation.
The [architecture tour](https://github.com/hebo1221/nano-deepseek-v4/blob/main/docs/architecture.md)
maps report sections to the readable native implementation and lists the
package's stable source surfaces.

## What it does not do

- It does not serve the official Pro 1.6T checkpoint at production latency.
- It does not provide frontier-scale DualPipe, fine-grained EP, multi-node NCCL,
  or production FP4 kernel infrastructure.
- It does not load or serve the official 0731 DSpark speculative decoder. The
  CPU checkers cover bounded draft equations and scheduler arithmetic on
  synthetic inputs, not that checkpoint's calibrated confidence, acceptance,
  kernels, cache layout, throughput, or quality.
- Its streaming checkpoint path is a verifier, not a sharded inference runtime.
- It does not reproduce the official published benchmark numbers.

It provides an executable interpretation of the major components so they can
be read, modified, and tested with explicit evidence boundaries.

## Citing

Please cite the DeepSeek-V4 report itself and use
[`CITATION.cff`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/CITATION.cff)
for this implementation.

## Project policies

Report reproducible failures through the
[structured bug form](https://github.com/hebo1221/nano-deepseek-v4/issues/new?template=bug_report.yml),
ask usage questions in
[Q&A](https://github.com/hebo1221/nano-deepseek-v4/discussions/categories/q-a),
and propose scoped features in
[Ideas](https://github.com/hebo1221/nano-deepseek-v4/discussions/categories/ideas).
Suspected vulnerabilities belong in the
[private security form](https://github.com/hebo1221/nano-deepseek-v4/security/advisories/new),
not a public issue.

See
[`PRODUCTION_READINESS.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/PRODUCTION_READINESS.md),
[`CONTRIBUTING.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/CONTRIBUTING.md),
[`SECURITY.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/SECURITY.md),
and [`CHANGELOG.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/CHANGELOG.md).
Maintainers can follow
[`RELEASING.md`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/RELEASING.md)
for the provenance-attested PyPI release path.

## License

Apache-2.0. See
[`LICENSE`](https://github.com/hebo1221/nano-deepseek-v4/blob/main/LICENSE).
