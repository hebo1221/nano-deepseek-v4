# Train and generate

This guide covers the fixed CPU reproduction command, the dependency-free
byte-text trainer, and the generation CLI. They use the complete tiny hybrid
stack, not a reduced dense surrogate, while keeping the default runs small
enough for CPU integration testing.

Install the current source from the repository root using the
[PyTorch-first source procedure](installation.md#cpu--current-source). For an
accelerator, select its PyTorch build before installing the package as described
in the same guide.

## Verify the installed learning path

Start with the fixed, wheel-bundled reproduction contract:

```bash
nano-deepseek-v4 reproduce
```

The command trains the fixed 246,590-parameter model for 20 CPU steps, stages a
native v2 bundle, reloads it, and verifies its complete checksum and tokenizer
inventory. With no destination, that staging bundle is removed after
verification. Retain a passing bundle and a machine-readable receipt when you
need to inspect or generate from the exact artifact:

```bash
nano-deepseek-v4 reproduce \
  --save-directory ./checkpoints/tiny-text \
  --output reports/tiny-text-reproduction.json \
  --json
```

`--save-directory` must be absent or an empty, non-symlink directory. Keep
`--output` outside the bundle directory tree; neither path may contain the
other.

The exit status and `acceptance` section use portable checks: fixed corpus,
configuration, optimizer and step recipe, finite losses, a validation-loss
reduction greater than 2.0, and an independently verified generation-ready
bundle. The `historical_observations` section is informational. For a registered
implementation it compares exact losses, sample bytes, tokenizer digest, and
bundle digest with a checked-in prior run, but those exact values are not a
cross-machine acceptance requirement.

### Reproduction exit codes

| Exit code | Meaning |
| --- | --- |
| `0` | The run completed and every portable acceptance check passed. A requested bundle was published. |
| `1` | The run completed, but at least one portable acceptance check failed. A requested bundle was not published. |
| `2` | `argparse` rejected the command line or the save/output path preflight. |
| `4` | A bundled contract or resource, execution, report serialization, or receipt write failed. |

Receipt writing happens after the run. An `output_write_failed` exit `4` can
therefore occur after a passing bundle was published; inspect the `artifact`
field when a JSON report is available instead of treating every `4` as an
unpublished bundle.

The command publishes `--save-directory` atomically only after portable
acceptance passes. The receipt's claim is limited to learning the fixed tiny
byte corpus and persisting a valid native bundle; it is not evidence of official
training parity or language quality.

## Customize the CPU learning run

Use the trainer directly when the fixed reproduction recipe is not the
experiment you want to run:

```bash
nano-deepseek-v4-train \
  --device cpu \
  --save-directory ./checkpoints/tiny-text
```

The deterministic 246,590-parameter configuration exercises sliding attention,
CSA, HCA, hash and learned MoE routing, MTP, Muon, and AdamW. The default 20
steps train on the built-in architecture corpus and produce output with this
shape:

```text
nano-deepseek-v4 tiny-text training
source: built-in architecture corpus
config: tiny_text_config (246,590 parameters)
device: cpu
validation loss: 7.2402 -> 4.5585
trained: 2,560 tokens in ...
bundle: checkpoints/tiny-text
bundle format: v2
bundle manifest sha256: ...
tokenizer sha256: ...
bundle round-trip: True
tokenizer round-trip: True
sample: ...
```

Timing is host-dependent and deliberately excluded from the acceptance
contract. The checked-in
[`tiny-text-training-baseline.json`](../../references/tiny-text-training-baseline.json)
also supplies the installed reproduction contract: it binds the fixed corpus,
configuration, implementation-specific history, acceptance threshold, and prior
observations without making exact historical bytes a portable pass condition.

Use your own raw UTF-8 or byte text, request a longer run, or capture the full
structured receipt:

```bash
nano-deepseek-v4-train --text-file README.md --steps 100
nano-deepseek-v4-train \
  --steps 20 \
  --output reports/tiny-text-training.json \
  --json
```

`--output` writes the complete JSON receipt through a temporary sibling and an
atomic rename, so a completed long run does not depend on terminal scrollback.

`device=auto` uses CUDA when available. A custom config must use the same
byte-v1 vocabulary and special-token IDs if the resulting bundle is expected to
be generation-ready.

## Generate from the saved bundle

The trainer passes the exact `ByteTokenizer` to `save_pretrained`, so the output
is a native v2 bundle. Its tokenizer mapping is included in the same checksum
inventory as the config, index, and weights.

```bash
nano-deepseek-v4-generate \
  --bundle ./checkpoints/tiny-text \
  --prompt "DeepSeek V4 " \
  --max-new-tokens 40 \
  --device cpu
```

Greedy decoding is the default. Reproducible sampling requires an explicit
temperature and seed:

```bash
nano-deepseek-v4-generate \
  --bundle ./checkpoints/tiny-text \
  --prompt "DeepSeek V4 " \
  --max-new-tokens 40 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 42 \
  --json
```

The JSON result retains the continuation token IDs because arbitrary generated
bytes need not form valid UTF-8. It also records the manifest, config, and
tokenizer digests used for the run.

Fixed-length generation does not stop on EOS by default. The raw-byte trainer
does not insert EOS targets into its continuous corpus stream, so an EOS default
would not reflect the training contract. Use `--stop-on-eos` only for a bundle
whose training data assigned that meaning to token 2.

## Generate from your own Hugging Face model repository

There is no first-party trained bundle claimed by this checkout. The Hub path
below is for a native bundle that **you** uploaded with the files from one v2
bundle directory.

Install the Hub client extra after selecting the PyTorch build:

```bash
python -m pip install -e ".[official]"
```

Then identify your model repository and an explicit tag, branch, or commit:

```bash
nano-deepseek-v4-generate \
  --hf-repo YOUR_USER/YOUR_NATIVE_BUNDLE \
  --revision YOUR_TAG_OR_COMMIT \
  --prompt "DeepSeek V4 " \
  --max-new-tokens 40
```

The command resolves the requested revision once, requires the returned
immutable commit identity, downloads the native manifest first, and accepts only
its bounded config/index/tokenizer/shard inventory. Every declared SHA-256 is
verified before model allocation. It does not execute repository code or call
Transformers auto classes.

The standard Hub cache represents snapshot files as symlinks to shared blobs,
while native bundle loading intentionally rejects shards that resolve outside
the bundle directory. After verification, this command therefore materializes
an isolated regular-file copy under the cache. `--cache-dir` controls that
location. Plan for roughly one additional bundle-sized copy on disk; this keeps
the native path-containment check intact instead of weakening it for Hub files.

The resolved commit and raw manifest digest identify the exact remote bundle in
the JSON output. Checksums detect corruption and incomplete copies; they do not
authenticate the publisher. Use a repository and revision you trust.

## Claim boundary

The default corpus is intentionally tiny and the tokenizer is byte-level. A
decreasing validation objective, deterministic generation receipt, and exact
save/load round trip establish that the implementation learns and persists its
own small experiment. They do not establish useful language quality or reproduce
official DeepSeek-V4 training.

For a real-corpus example and one matched control, continue with the
[Tiny Shakespeare control](../research/tiny-shakespeare.md).
