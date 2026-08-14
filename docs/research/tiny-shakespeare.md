# Tiny Shakespeare training and attention control

This study asks two bounded questions:

1. Can the 8.50M-parameter `mini` preset learn a real byte-level corpus and
   survive an exact native-bundle round trip?
2. Under otherwise matched 128-token eager-PyTorch training, does replacing the
   hybrid attention schedule with all sliding attention produce a detectable
   quality or throughput difference?

It does not reproduce official DeepSeek-V4 benchmarks and does not support an
architecture-superiority claim.

## Corpus

Download Tiny Shakespeare at the exact `karpathy/char-rnn` revision used by the
receipts, then verify its bytes:

```bash
mkdir -p data/tiny-shakespeare
curl --fail --location \
  https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt \
  --output data/tiny-shakespeare/input.txt
python - <<'PY'
from hashlib import sha256
from pathlib import Path

path = Path("data/tiny-shakespeare/input.txt")
expected = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
actual = sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
print(f"SHA-256 OK: {path}")
PY
```

The receipt binds 1,115,394 bytes and 40,000 lines to that revision and SHA-256
digest.

## Hybrid baseline

The reference run used:

```bash
nano-deepseek-v4-train \
  --text-file data/tiny-shakespeare/input.txt \
  --model-preset mini \
  --steps 5000 \
  --context-length 128 \
  --batch-size 4 \
  --eval-batches 8 \
  --learning-rate 3e-4 \
  --seed 1337 \
  --device cuda \
  --max-new-tokens 300 \
  --prompt $'\nROMEO:' \
  --temperature 0.8 \
  --top-p 0.9 \
  --save-directory checkpoints/tiny-shakespeare-5000 \
  --output reports/tiny-shakespeare-training.json \
  --json
```

Observed on one NVIDIA GB10 with PyTorch `2.13.0+cu130`:

| Measurement | Observed value |
| --- | ---: |
| Parameters | 8,502,150 |
| Validation loss | 7.2491 to 1.9757 |
| Trained tokens | 2,560,000 |
| Elapsed time | 480.86 s |
| Throughput | 5,324 tokens/s |
| Native bundle round trip | exact match |

The reported objective is `causal LM cross-entropy + 0.3 * MTP
cross-entropy`. It is not perplexity. The complete receipt is
[`../../references/tiny-shakespeare-training-baseline.json`](../../references/tiny-shakespeare-training-baseline.json).

The checked-in GB10 training receipts are historical artifacts from result
schema 2 and implementation digest
`ee41cd36f812dcf683986de6e7fb45bd200c65cbad7268acd5a88e58da156584`;
their saved checkpoints used model-only v1 bundles. The current trainer emits
result schema 3 and tokenizer-bound v2 bundles.

The study receipt binds each raw run by SHA-256. Corpus, configuration, and
bundle identities are recorded separately, so machine-local output locations
are not part of the reproduction procedure. These same-machine results are
deterministic-replay evidence, not a promise of byte-identical training across
hardware or PyTorch versions.

The seeded sample began:

```text
ROMEO:
What are you not my mother be to kill and hear to
conscience did say 'tis a gave the noble before.

SICINIUS:
A gentleman buckious that?
```

This is evidence that the training path learns corpus structure; the sample is
not a language-quality benchmark.

## Matched all-sliding control

The control changes the three-layer attention schedule from sliding / CSA / HCA
to sliding / sliding / sliding. Corpus bytes, split, random seeds, sampled
windows, optimizer settings, MoE and MTP schedules, tokenizer, generation
settings, and all other training arguments remain matched.

```bash
nano-deepseek-v4-train \
  --text-file data/tiny-shakespeare/input.txt \
  --config references/tiny-shakespeare-all-sliding-control-config.json \
  --steps 5000 \
  --context-length 128 \
  --batch-size 4 \
  --eval-batches 8 \
  --learning-rate 3e-4 \
  --seed 1337 \
  --device cuda \
  --max-new-tokens 300 \
  --prompt $'\nROMEO:' \
  --temperature 0.8 \
  --top-p 0.9 \
  --save-directory checkpoints/tiny-shakespeare-all-sliding-5000 \
  --json
```

The original matched pair produced:

| Variant | Parameters | Final 8-batch validation loss | Throughput |
| --- | ---: | ---: | ---: |
| Hybrid | 8,502,150 | 1.9757 | 5,324 tokens/s |
| All sliding | 8,440,758 | 1.9644 | 5,630 tokens/s |

The eight-batch training-end summaries are too small for a quality conclusion.
The paired evaluator therefore loads both checksum-verified bundles and
evaluates them on the same 256 held-out byte windows:

```bash
nano-deepseek-v4-compare \
  --text-file data/tiny-shakespeare/input.txt \
  --left-bundle checkpoints/tiny-shakespeare-5000 \
  --right-bundle checkpoints/tiny-shakespeare-all-sliding-5000 \
  --left-label hybrid \
  --right-label all_sliding \
  --batches 256 \
  --batch-size 4 \
  --context-length 128 \
  --validation-seed 1338 \
  --device cuda \
  --output reports/tiny-shakespeare-attention-control-evaluation.json \
  --json
```

For seed 1337, the paired hybrid-minus-all-sliding total-loss difference was
`+0.00416`, with a descriptive normal-approximation interval of
`[-0.00009, +0.00842]`. The interval crosses zero. Randomly sampled byte windows
overlap, so this is a descriptive interval, not an inferential confidence
interval over independent observations. It also does not measure
training-seed variation.

The control design and paired evaluation are recorded in:

- [`../../references/tiny-shakespeare-attention-control.json`](../../references/tiny-shakespeare-attention-control.json)
- [`../../references/tiny-shakespeare-attention-control-evaluation.json`](../../references/tiny-shakespeare-attention-control-evaluation.json)

## Claim boundary

- One training seed does not characterize training-seed variation or support an
  architecture-superiority claim.
- Held-out byte windows overlap; within-run intervals are not independent-sample
  confidence intervals.
- Throughput applies to this eager PyTorch implementation, 128-token contexts,
  and one NVIDIA GB10. It is not evidence about long contexts, optimized
  attention kernels, or other hardware.
- These runs do not reproduce the official DeepSeek-V4 benchmark numbers.

Return to the [`research index`](README.md) or inspect the
[`training and generation guide`](../guides/train-and-generate.md).
