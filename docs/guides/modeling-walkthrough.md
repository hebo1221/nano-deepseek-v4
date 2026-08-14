# Read the model in ten minutes

The fastest way to understand this repository is to follow one real tensor
through the tiny model:

```bash
nano-deepseek-v4 tour
```

The command runs on CPU, downloads nothing, and reports shapes observed by
hooks on the actual modules. It does not maintain a second explanatory model.
Use `--json` when you want the same trace as structured data.

The fixture has eight input tokens and the normal four-layer tiny schedule:
two sliding-attention layers, one Compressed Sparse Attention (CSA) layer, and
one Heavily Compressed Attention (HCA) layer. Its compression rates are shortened
to 2:1 and 4:1 so both compressed paths produce visible slots in eight tokens.
That change is educational only; it is not an official Flash configuration.

## The complete path

```text
token ids [1, 8]
    |
    +-- token embedding [1, 8, 64]
    +-- copy into four mHC residual streams [1, 8, 4, 64]
    +-- layer 0: sliding attention + hash MoE
    +-- layer 1: sliding attention + hash MoE
    +-- layer 2: CSA + Lightning Indexer + hash MoE
    +-- layer 3: HCA + learned top-k MoE
    +-- HyperHead collapse [1, 8, 64]
    +-- RMSNorm + LM head [1, 8, 512]
    +-- MTP depth 1 with shifted embeddings [1, 7, 512]
```

These dimensions are small, but the module boundaries are the same ones used
by the larger native presets. The official 0731 checkpoint adds a separate
DSpark attachment; it is inspectable but is not routed through this native
forward pass.

## 1. Embeddings become mHC streams

Start in
[`DeepSeekV4Model._forward_with_streams`](../../nano_deepseek_v4/modeling.py).
`embed_tokens` maps token IDs to one hidden vector per position. The model then
copies each vector into `hc_mult=4` residual streams.

Every attention and feed-forward sublayer is wrapped by `HyperConnection`. It
computes `pre` weights to collapse the residual streams into the sublayer
input, then returns three objects:

- `post`: how strongly the sublayer output enters each stream;
- `comb`: a Sinkhorn-normalized matrix that mixes the existing streams;
- `collapsed`: the `pre`-weighted input passed to the sublayer.

The sublayer receives one collapsed `[batch, tokens, hidden]` tensor. Its result
is written back into all four streams, so the outer shape stays
`[1, 8, 4, 64]` from the first decoder layer to the last.

Read next: `HyperConnection`, `HyperHead`, then
`DeepSeekV4DecoderLayer.forward` in
[`modeling.py`](../../nano_deepseek_v4/modeling.py).

## 2. One attention class, three memory policies

`DeepSeekV4Attention` owns the common query, shared K=V, partial-RoPE, attention
sink, and grouped output paths. Its `layer_type` changes how older memory joins
the local sliding window.

### Sliding attention

Layers 0 and 1 use only the most recent four keys in this fixture. The mask is
both causal and windowed. With caching, `DeepSeekV4LayerCache.update_local` retains the complete observed
KV history so speculative rollback remains possible; the attention mask still
limits each query to the configured window.

### Compressed Sparse Attention

Layer 2 uses `CSACompressor` at 2:1. Eight positions form four compressed slots.
`CSAIndexer` builds its own compressed index, scores those entries for each
query, applies causality, and selects at most two entries. The tour prints the
observed range `0..2`: the first query cannot see a completed past compression
window, while later queries can see two.

The selected entries are concatenated with local keys before the shared
attention calculation. CSA is sparse retrieval over compressed memory, not a
replacement for the local window.

### Heavily Compressed Attention

Layer 3 uses `HCACompressor` at 4:1, so eight positions form two compressed
slots. Unlike CSA, HCA does not run a top-k indexer: every causally available
compressed slot joins the local keys.

Read `DeepSeekV4Attention.forward`, then `CSACompressor`, `CSAIndexer`, and
`HCACompressor` in [`modeling.py`](../../nano_deepseek_v4/modeling.py).

## 3. Hash routing hands off to learned routing

Every decoder layer contains `DeepSeekV4MoE`. The first three tiny layers use
the hash bootstrap path: token IDs select two routed experts through a static
`tid2eid` table. The fourth layer uses learned top-k routing over
`sqrt(softplus(logits))` affinity scores. Both paths also add the configured shared
expert. Hash routing fixes the expert IDs through `tid2eid`, but their mixture
weights still come from the gathered `sqrt(softplus(logits))` affinities before
normalization and scaling.

The tour reports router logits with shape `[1, 8, 8]`: one score for each of
eight routed experts at every token. It describes the routing rule, not the
quality or specialization of randomly initialized experts.

Read `DeepSeekV4MoE.route`, `DeepSeekV4MoE.forward`, and `SwiGLUExpert`.

## 4. The head and MTP branch

After the backbone, `HyperHead` collapses the four residual streams and the
main RMSNorm produces `[1, 8, 64]`. The shared language-model head maps that to
512 vocabulary logits per position.

Supplying labels also activates `DeepSeekV4MTPModule`. At depth one it combines
the backbone streams shortened to seven positions with embeddings shifted one
token into the future. The MTP block emits `[1, 7, 512]` logits through the same
LM head. This native MTP depth is distinct from the three-stage DSpark metadata
found in the 0731 checkpoint.

Read `DeepSeekV4ForCausalLM._mtp_forward` and `DeepSeekV4MTPModule`.

## 5. Why the command checks the cache

The tour finishes by comparing one eight-token forward pass with a five-token
prefill followed by a three-token cached suffix. A pass means that the local,
CSA, and HCA cache states reproduce the full-forward logits within the fixed
numeric tolerance.

That is a correctness check for this tiny path. It says nothing about text
quality, official weights, production kernels, or accelerator speed.

## Explore without allocating the official model

Once the tiny path is clear, inspect how the same families are scheduled in the
large presets:

```bash
nano-deepseek-v4 inspect --preset flash
nano-deepseek-v4 inspect --preset flash-0731 --json
```

The inspection commands count structure without constructing the model. The
0731 report separates 43 backbone layers from its three DSpark stages and marks
the official-checkpoint runtime as unsupported.

Useful next files:

- [`config.py`](../../nano_deepseek_v4/config.py): schedules, dimensions, and
  official configuration ingestion;
- [`architecture.md`](../architecture.md): report-to-code map and package tour;
- [`attention-reach.md`](../verification/attention-reach.md): a structural test
  showing which source positions can influence a later query;
- [`transformers-parity.md`](../verification/transformers-parity.md):
  differential equation checks against the pinned independent implementation.

## Change one thing at a time

For a small experiment, construct `DeepSeekV4Config` with one changed field and
run the normal model API. Good first changes are `sliding_window`, `index_topk`,
or one compression rate. Keep the sequence long enough to complete at least one
compression window.

Use the tour as a map, not as a benchmark. If a change affects equations, add a
focused unit test and run the relevant conformance or parity check before
drawing a conclusion.
