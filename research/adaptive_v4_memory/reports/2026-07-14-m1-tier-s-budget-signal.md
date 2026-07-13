# M1 Tier-S budget-signal report

Status: the M1 predictive gate passed at both tested scales; no model-quality or
systems claim is eligible.

## Scope

Two BF16 models were trained on generated associative recall using answer,
indexer-ranking, compressed-attention read-ranking, and query-conditioned block
value losses. The models contain 55,387,476 and 151,352,716 inference
parameters. Training-only auxiliary decoders are excluded from checkpoints.
Raw tokens and checkpoints are not committed; their sizes and SHA-256 digests
are recorded in the checked
[`summary`](../results/m1-tier-s-budget-signal.summary.json).

The first three training designs failed at random accuracy: causal key/value
order was wrong in the first run, then the discrete top-k indexer received no
gradient, and finally full-memory single-query training converged to a uniform
value solution. These failures motivated the differentiable CSA probe and the
three explicit memory-path objectives. Failed checkpoints remain ignored local
artifacts and are not used in the reported evaluation.

## Retrieval evidence

At the best recorded step, S55 reached 74.1% native top-8 accuracy versus 1.3%
local-only; S151 reached 82.3% versus 1.3%. Both missed the separately fixed
85% training target. The failures are retained rather than changing the target
after observing results.

On 768 deterministic held-out examples per scale, S55 exact-match accuracy was
2.5%, 52.1%, 64.7%, 72.3%, and 72.9% at top-k 0, 1, 2, 4, and 8. S151 reached
1.3%, 69.1%, 73.4%, 76.3%, and 76.7%. This establishes that both models use
remote compressed memory and that most successful cases need few blocks.

## Predictive gate

The label is the minimum shared per-CSA-layer top-k that produces an exact
answer, with a dense-fallback label when no tested budget succeeds. Validation
holds out each generated example as one group, preventing its layer rows from
leaking into training.

- S55: context-blocks + layer baseline MAE 5.298; full native features MAE
  4.667; 11.9% improvement.
- S151: baseline MAE 4.498; full native features MAE 3.767; 16.3% improvement.

Both exceed the preregistered 5% improvement threshold, so M2 training-free
controller work may proceed. This is not a positive research claim: only one
training seed and one synthetic task were run, and both scales missed the 85%
training target.
