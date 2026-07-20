# HCA--CSA conditional-sufficiency H0 protocol

Status: instrumentation and prerequisite design, 2026-07-20. This protocol
does not authorize a residency, offload, prefetch, or quality claim.

## 1. Question

For a same-token HCA layer followed by a later CSA layer, does the HCA read
path contain information about the later native CSA decision that is not
already available from:

1. the target CSA layer at the preceding token;
2. the nearest causally earlier CSA layer at the same token;
3. recency, locality, and query shift; and
4. a matched-cost first-order block descriptor?

The primary candidate is not generic HCA-guided prefetch. It is the narrower
claim that HCA can improve a causal anchor, detect when that anchor is unsafe,
or decide when a second-order descriptor is worth reading.

## 2. Direct-threat audit

The following primary papers determine the minimum comparison set.

- **IndexCache** (arXiv:2603.12201) makes nearest-prior CSA selection reuse a
  mandatory baseline. Adjacent top-k overlap is often high, but the paper's
  negative result shows that overlap and local output cosine similarity do not
  identify safe reuse patterns; small missed critical sets propagate. We must
  measure end-to-end loss or causal output damage, not overlap alone.
- **ECHO** (OSDI 2026) predicts the next top-k threshold from score history and
  performs early transfers while retaining a guaranteed exact recall. HCA must
  add value beyond prior-token score/threshold history, and any later systems
  comparison must preserve the final native top-k reference.
- **COBS** (arXiv:2607.09052) identifies DeepSeek-V4 CSA gated pooling as a
  first-order selector and attributes its error to omitted within-block
  covariance. HCA is a different learned path, not a block mean; COBS theory
  cannot be assigned to HCA without measurement. A fixed second-order
  descriptor is the required alternative if HCA has residual signal.
- **FlashMemory-DeepSeek-V4** (arXiv:2606.09079) already trains a three-layer
  lookahead indexer over frozen native compressed keys. Its reported memory
  reductions therefore invalidate novelty claims based only on predictive
  retrieval. Its MRCR failure, context-independent false-positive growth, and
  length-generalization ceiling make dense-query detection and fallback the
  relevant open boundary.
- **LiteTopK** (2026) removes score-logit materialization while preserving
  exact TopK and reports that the indexer/TopK path can dominate long-prefill
  runtime. A future HCA method must count descriptor materialization and beat
  this exact fused-selector systems baseline; offline FLOP estimates are not
  enough.

## 3. Fail-closed prerequisite discovered

The preserved Tier-S checkpoints cannot test this H0 directly.

| Audit | s55 | s151 |
| --- | ---: | ---: |
| training lengths | 64, 80 | 64, 80 |
| evaluation lengths | 48, 64, 80 | 48, 64, 80 |
| HCA compression rate | 128 | 128 |
| HCA tensors exactly equal to seeded initialization | 12/12 | 20/20 |
| CSA tensors changed from seeded initialization | 30/30 | 50/50 |

At fewer than 128 tokens, `HCACompressor` produces no compressed entry. The
HCA parameters consequently receive no task gradient. The equality audit
reconstructed each model with its recorded initialization seed and compared
the checkpoint tensors with `torch.equal`.

Therefore:

- an HCA result from the existing checkpoints is an **untrained-path negative
  control**, never primary evidence;
- a zero HCA residual on those checkpoints cannot falsify H0; and
- H0 execution requires a separately identified long-context continuation
  checkpoint. The original checkpoints and frozen P2 artifacts remain
  unchanged.

The old data geometry has a second confound: all key/value pairs occupy the
first 31 positions, so even a longer padded sequence places every evidence
item in one HCA span. H0 training and evaluation must distribute evidence
across independently sampled 128-token spans.

## 4. Minimal observation contract

A research-only sidecar observes the compressor outputs and attention-core
inputs already passed by the real evaluation path, then recomputes diagnostic
scores and probabilities under `no_grad`. It installs temporary instance hooks,
does not modify the package implementation, returns the original attention
result unchanged, and restores every hooked instance on exit. Normal forwards
do not record or compute these observations.

For every HCA and CSA read:

- layer index and memory type;
- query positions and compressed-entry end positions;
- per-head masked read score;
- per-head normalized attention probability, including competition with local
  memory and the attention sink; and
- per-head pre-output-projection contribution L2.

The existing CSA probe additionally retains every native indexer score,
native top-k membership, compressed value, and dense pre-selection read score.
HCA spans align to a CSA entry only when the CSA end position lies in
`(hca_end - 128, hca_end]`. Boundary and causal validity are derived from the
recorded positions, never from row order.

For target CSA layer `l` and query `q`, only signals available before `l` may
be features:

- target layer `l` at query `q-1`;
- nearest earlier CSA layer at query `q`;
- nearest earlier HCA layer at query `q`; and
- metadata known before the target indexer is issued.

Target-layer query-`q` scores, selections, attention mass, and patch damage are
labels only. A digest-bound derived dataset must reject absent positions,
non-increasing layer order, duplicate keys, incomplete HCA spans, and any
target feature leakage.

## 5. HCA-aware continuation prerequisite

The first pilot initializes from one preserved checkpoint per scale and writes
new artifacts under a new experiment ID. It does not mutate or relabel the
source checkpoints.

- train on 512- and 640-token distributed associative-recall sequences;
- place evidence across at least three completed HCA spans per sequence;
- keep tail queries outside the local window and after those completed spans;
- retain the existing answer, CSA index-ranking, CSA read-ranking, and
  query-conditioned value objectives;
- add **no HCA-specific ranking label or HCA selection loss**, because directly
  supervising the proposed signal would make the H0 test circular; and
- evaluate native, local-only, and full-memory behavior on held-out generators.

The continuation is admissible for H0 only if:

1. source checkpoint/config/digest and continuation seed are recorded;
2. HCA parameters change and receive finite nonzero gradients;
3. every evaluated query has at least three causally available HCA spans;
4. evidence spans are non-degenerate across the held-out set;
5. native CSA remains meaningfully above local-only; and
6. HCA attention is neither identically zero nor a single constant-span rule.

Failure is a training/data-prerequisite failure, not an H0 result.

## 6. H0 models and targets

The primary target is membership in the later native CSA top-k. Secondary
targets are native CSA attention mass, complete top-k hit, and causal damage
from frozen block-drop replay measured by output KL, target loss delta, and
top-logit change.

Nested held-out models are fit on whole-conversation splits:

| Model | Features |
| --- | --- |
| B0 | recency/locality only |
| B1 | B0 + prior-token target-layer score history (ECHO-style) |
| B2 | B1 + nearest prior-CSA scores/selections (IndexCache-style anchor) |
| H | B2 + HCA score, probability, contribution, and anchor disagreement |
| S1 | B2 + matched-byte first-order descriptor |
| S2 | B2 + fixed low-rank second-order descriptor (COBS-style) |
| N | unmodified native indexer; semantic reference, not a predictor baseline |

All candidate rankings are evaluated at the same expanded CSA-block budget.
Because one HCA span contains approximately 32 four-token CSA entries, HCA may
not claim a win by expanding every high-scoring span without charging all 32
entries and their metadata.

## 7. Pilot and confirmatory decisions

The two-scale pilot is diagnostic. It verifies identifiability, feature
variance, split integrity, and approximate effect size without a publication
claim. Candidate thresholds are frozen only after a baseline-only view.

HCA proceeds beyond trace-only study only if model H improves the prespecified
held-out top-k recall or causal-damage deviance over B2 on both scales and its
seed/conversation-clustered 95% interval excludes zero in the confirmatory
cohort. Coarse expansion must remain at most four times native top-k, with at
least 99% mean top-k recall, at least 97% in every major family, and at least
95% complete-hit queries.

If H fails, HCA is rejected as a directory. If H helps only as a damage
sentinel, the project narrows to fallback acquisition. If S2 dominates H at
matched bytes, the project becomes selective second-order acquisition. No
physical prefetch implementation begins before one of those outcomes and a
separate causal timing gate.
