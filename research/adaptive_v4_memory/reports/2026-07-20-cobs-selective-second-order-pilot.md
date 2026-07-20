# Selective second-order CSA pilot

Date: 2026-07-20

Decision: **do not promote the descriptor or acquisition gate to a runtime or
multi-seed experiment.** A current-token second-order statistic contains real
signal, but the causal prior-token version does not reproduce on both scales
and the current-token version does not beat the exact native indexer at the
operational one- or two-times block budgets.

This is a single-continuation-seed synthetic diagnostic. It is not evidence
about official DeepSeek-V4 weights or natural-language quality.

## Question

The HCA directory pilot left one narrower alternative:

> Can B2 uncertainty identify the queries for which a compact COBS-style
> second-order descriptor is worth reading?

COBS (arXiv:2607.09052) derives the second-order approximation

`log mass ~= log block_size + q^T mean + 0.5 q^T covariance q`.

The relevant direct threat is stronger than native-selection overlap. The
descriptor must help recover blocks that matter under a dense attention
counterfactual, at matched block and descriptor budgets, and must ultimately
justify its later arrival relative to the causal B2 prefetch anchor.

## Frozen diagnostic design

The pilot reused the HCA-aware s55 and s151 continuation checkpoints and the
same deterministic distributed associative-recall generator. It did not
modify a checkpoint, the package, or frozen P2 artifacts.

- lengths: 512 and 640;
- 20 traces per length, batch 2, four tail queries per item;
- primary fit/test split: 15/5 whole traces per length;
- gate split: 10 base-fit / 5 gate-calibration / 5 test traces per length;
- layer-stratified regularized logistic models for native-selection labels;
- layer-stratified ridge models for log block-drop damage;
- 5,000 trace-cluster bootstrap resamples for selection log-loss deltas.

For each CSA entry, the diagnostic reconstructed the compressor's eight
overlapping source slots, excluded the synthetic zero predecessor slots in the
first entry, applied the compressor RMS normalization and block RoPE, and
projected the slots into an eight-dimensional query subspace fit only on the
training split. It then stored:

- S1 mean: 8 FP4 values, 4 bytes per block;
- S2 symmetric covariance: 36 FP4 values, 18 bytes per block;
- S1+S2 total: 22 bytes per block;
- reference CSA K=V entry in this Tier-S model: 64 BF16 values, 128 bytes.

`S1` and `S2` use target-layer query `q-1` and are causal prefetch candidates.
`T1` and `T2` use current query `q`; they are timing upper bounds, not early
prefetch features. B2 retains recency, prior-token target score, query shift,
and nearest-prior-CSA score/selection.

## Causal damage label

Native block-drop damage alone is biased for this question: every block that
the native indexer did not select has exactly zero native-path drop damage.
The terminal diagnostic therefore also removes the sparse mask, retains local
and causal masks plus the attention sink, and computes a dense-CSA
counterfactual over every historical compressed entry.

For each entry with attention probability `p`, value `v`, and dense context
`c`, the post-drop context is `(c - p*v) / (1-p)`. The resulting delta is sent
through inverse RoPE and the real grouped/output projections. Its output L2 is
the dense local block-drop label. A direct `CSASelectionPlan` intervention
validated the analytic value:

| Validation | Analytic | Actual plan | Relative error |
| --- | ---: | ---: | ---: |
| native-path largest drop | 40.6799 | 40.6069 | 0.180% |
| dense-path largest drop | 31.1397 | 31.1308 | 0.029% |

This is an exact layer-output intervention up to BF16 recomputation error. It
is not a final-logit or answer-loss intervention.

## Selection signal

The prior-token second-order increment does not reproduce. The current-token
increment does.

| Held-out log-loss improvement | s55 | s151 |
| --- | ---: | ---: |
| S1 - B2 | +0.000867 `[+0.000412,+0.001335]` | -0.000086 `[-0.000237,+0.000050]` |
| S2 - B2 | +0.001014 `[+0.000559,+0.001457]` | +0.000076 `[-0.000226,+0.000382]` |
| S2 - S1 | +0.000148 `[-0.000005,+0.000297]` | +0.000163 `[-0.000064,+0.000393]` |
| T2 - B2 | +0.002910 `[+0.001816,+0.003820]` | +0.004264 `[+0.003215,+0.005231]` |
| T2 - T1 | +0.002352 `[+0.001533,+0.003149]` | +0.001517 `[+0.001087,+0.001926]` |

Thus the covariance term is informative only when the current target-layer
query is already available. That finding does not authorize prefetch: it has
the same timing class as the native indexer.

## Dense counterfactual damage recall

The table ranks historical blocks and reports the fraction of total dense
local block-drop damage captured at 1x, 2x, and 4x the native top-8 budget.
`Native` ranks by the real target indexer score. `Oracle` ranks by the dense
damage label itself.

| Scale / ranker | 1x | 2x | 4x |
| --- | ---: | ---: | ---: |
| s55 B2 | 0.82581 | 0.91379 | 0.98210 |
| s55 T2 | 0.83009 | 0.92669 | 0.97907 |
| s55 Native | **0.88876** | **0.94636** | 0.97689 |
| s55 Oracle | 0.95761 | 0.98796 | 0.99604 |
| s151 B2 | 0.87864 | 0.92877 | 0.95970 |
| s151 T2 | 0.88095 | 0.93586 | 0.96207 |
| s151 Native | **0.88818** | **0.94235** | **0.96750** |
| s151 Oracle | 0.92344 | 0.96691 | 0.98637 |

T2 improves B2 at 1x and 2x on both scales, but the exact native indexer is
better at those same budgets on both scales. At 4x, T2 is mixed: below B2 on
s55 and below Native on s151. There is no two-scale matched-budget dominance.

## Selective acquisition gate

A linear gate saw only B2 query-level uncertainty features: probability
moments, normalized entropy, boundary margin, top-budget mass, query shift,
prior-CSA selected rate, layer, position, and candidate count. It predicted
whether reading T2 for a query would improve 2x-budget dense-damage recall.

| Scale / policy | Descriptor acquisition | Dense-damage recall | Gain over B2 |
| --- | ---: | ---: | ---: |
| s55 B2 only | 0% | 0.91291 | 0 |
| s55 top-half gate | 50% | 0.92647 | +0.01357 |
| s55 all T2 | 100% | 0.92725 | +0.01434 |
| s55 oracle query gate | outcome-aware | 0.93214 | +0.01923 |
| s151 B2 only | 0% | 0.92884 | 0 |
| s151 top-half gate | 50% | 0.93453 | +0.00569 |
| s151 all T2 | 100% | 0.93568 | +0.00684 |
| s151 oracle query gate | outcome-aware | 0.93757 | +0.00873 |

The learned gain correlations were only 0.221 and 0.238. The gate can preserve
most of all-T2's small gain while avoiding half of descriptor acquisitions,
but it does not create a quality advantage over the native indexer.

## Gate decision

1. **Causal second-order prefetch: fail.** S2 over S1 is not positive on both
   scales and does not produce a two-scale damage advantage.
2. **Same-token second order: bounded signal.** T2 improves B2 deviance and
   2x dense-damage recall, so the covariance is not empty information.
3. **Strong-baseline gate: fail.** Native indexer scoring remains better at
   the operational 1x and 2x budgets on both scales.
4. **Selective gate: insufficient.** It saves descriptor reads but does not
   reverse the strong-baseline result.
5. **Novelty/timing gate: fail.** COBS already establishes current-token
   second-order selection, and this pilot provides neither an earlier signal
   nor a measured systems advantage over exact fused/native selection.
6. **Runtime and confirmatory expansion: denied.** No five-seed run,
   final-logit intervention matrix, kernel, prefetch path, or HBM experiment is
   justified from this diagnostic.

The only admissible reconsideration would start with official V4 timing and
bandwidth evidence that a 22-byte descriptor can be read and acted on earlier
or materially cheaper than the native indexer while meeting a preregistered
quality non-inferiority bound. Without that evidence, effort returns to the
already authorized calibrated+pins versus fixed+pins P2 study.

## Artifact and implementation boundary

The exploratory driver grew to 1,038 lines while separating sparse and dense
counterfactuals, timing classes, and the three-way gate split. Promoting it
would violate the explicit 500-line additional-design stop rule. It therefore
remains an ignored local diagnostic and is not added to the package or research
runtime. This result cannot be promoted to confirmatory evidence.

| Local artifact | SHA-256 |
| --- | --- |
| exploratory driver | `475d2e5dd6bafac47181ebc8b99ca004a622abc27d43f63f4ea63707945045fb` |
| s55 summary | `2d17df9d8aaac1811d13caabe38718eec0649adf23525ad2462bb243005c2303` |
| s151 summary | `e1f4c536ca994cfed132517c9252aa968c38a681fce17468d6a279ce197c9ebd` |
| s55 derived-row digest | `d2c397dea0a594193674c2a3aba185d975fca0aeacfa9b746fe039f337b0a45b` |
| s151 derived-row digest | `56be1274c4f09b19f1971aa0ebaed7516c568ec31b9016cfe7f9d2d32173fa8a` |
