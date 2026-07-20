# HCA--CSA conditional-sufficiency H0 pilot

Date: 2026-07-20

Decision: **HCA directory NO-GO; retain HCA only as a diagnostic/sentinel
ablation and move the next pilot to prior-CSA plus selective second-order
evidence.**

This is a single-continuation-seed synthetic diagnostic. It is not a
confirmatory result and does not establish that HCA lacks useful information
in official DeepSeek-V4.

## Why this pilot was necessary

The primary-source audit made the old proposed comparison too weak:

- IndexCache requires nearest-prior CSA reuse as the anchor and shows that
  overlap/local similarity alone is not a quality proxy;
- ECHO makes prior-token score and threshold history a required causal
  baseline while preserving exact final top-k recall;
- COBS shows why a first-order cached selector can miss within-block
  curvature, making fixed second-order evidence the direct alternative; and
- FlashMemory-DeepSeek-V4 already implements learned predictive CSA retrieval,
  so generic HCA-guided prefetch is not a defensible novelty claim.

The checkpoint audit then found that the original Tier-S study could not test
HCA at all. Training and evaluation stopped at 80 tokens while the HCA rate is
128. Reconstructing each model from its recorded seed showed every HCA tensor
bitwise equal to initialization: 12/12 tensors for s55 and 20/20 for s151.
Every CSA tensor checked had changed (30/30 and 50/50). Existing checkpoints
are therefore untrained-HCA controls, not negative H0 evidence.

## Instrumentation result

A research-only sidecar records HCA and CSA read-path head scores, normalized
probabilities, and pre-output-projection contribution L2 by observing real
compressor outputs and recomputing diagnostics from attention-core inputs under
`no_grad`. It pairs only causally ordered
`prior CSA < HCA < target CSA` rows and keeps the target CSA score, selection,
mass, and margin as labels. The package and normal CSA probe remain unchanged.
The sidecar returns the original attention result, removes every temporary hook
on exit, and is rejected in training mode.

The dedicated sidecar regression passes **2/2** tests. Sidecar-on and
sidecar-off hidden states are bitwise equal before and after hook removal. The
two-scale re-extraction reproduced the original row digests, summaries, and
all reported B2/H statistics exactly.

## HCA-aware continuation prerequisite

One preserved seed per scale (`6071401`) was continued for 200 steps, batch 2,
at lengths 512 and 640. Evidence pairs were spread evenly over at least three
completed 128-token HCA spans. The objective retained answer loss and the
existing CSA index/read/value objectives. It added no HCA ranking target,
HCA selector loss, or other HCA-specific supervision.

| Prerequisite | s55 | s151 |
| --- | ---: | ---: |
| nonzero HCA-gradient steps | 200/200 | 200/200 |
| HCA parameter delta L2 | 1.0947 | 1.6835 |
| paired native accuracy, 512 | 1.0000 | 0.7422 |
| paired local-only accuracy, 512 | 0.0000 | 0.0234 |
| paired native accuracy, 640 | 0.9922 | 0.7812 |
| paired local-only accuracy, 640 | 0.0078 | 0.0234 |
| paired no-HCA accuracy, 512 / 640 | 1.0000 / 0.9922 | 0.7188 / 0.7500 |

HCA features had nonzero variance on both scales. The continuation prerequisite
therefore passed. The paired quality diagnostic suggests no direct HCA quality
lift at s55 and a small 2.3--3.1 percentage-point lift at s151; this is a
descriptive observation over 128 answers per length, not an inference result.

## Conditional-sufficiency design

The target was membership in a later layer's native CSA top-8 among historical
HCA-covered blocks. Whole traces, not candidate rows, defined the split. Each
length used 15 training and five test traces. The test set therefore contained
10 independent trace clusters across the two lengths.

The layer-stratified B2 model used:

- recency and query position;
- prior-token target-layer score;
- prior-query shift; and
- nearest-prior CSA score and selected membership.

H added HCA per-span score, probability, and contribution mean/max. Both arms
used the same regularized logistic estimator. Uncertainty was estimated with
5,000 trace-cluster bootstrap resamples. The alpha metrics rank only historical
HCA-covered blocks; uncovered local/tail blocks are outside this directory
comparison and would remain resident in a runtime design.

## Result

| Metric | s55 B2 | s55 H | s151 B2 | s151 H |
| --- | ---: | ---: | ---: | ---: |
| held-out log loss | 0.157630 | 0.156978 | 0.082971 | 0.082850 |
| AUROC | 0.89770 | 0.89881 | 0.98160 | 0.98167 |
| alpha=4 mean historical top-k recall | 0.86696 | 0.86607 | 0.99570 | 0.99570 |
| alpha=4 complete historical hit | 0.43125 | 0.40625 | 0.96875 | 0.96875 |

The mean H-over-B2 log-loss improvements were:

- s55: `+0.000643`, trace-bootstrap 95% interval
  `[+0.000386, +0.000891]`;
- s151: `+0.000116`, trace-bootstrap 95% interval
  `[-0.000015, +0.000276]`.

The effect was layer-local rather than uniform. At s55, almost all lift came
from target layer 6; layer 4 was effectively zero. At s151, layers 8 and 10
were positive while layers 4 and 6 were negative. This agrees with
IndexCache's warning that layer-local averages can hide critical transitions,
but it does not rescue the systems claim: H produced no alpha=4 recall or
complete-hit lift and failed the two-scale conditional interval gate.

## Gate decision

1. **Conditional signal: fail.** The s151 clustered interval crosses zero, so
   the required two-scale reproduction is absent.
2. **Selector recall: fail.** s55 reaches only 86.6% mean recall and 40.6%
   complete-hit at the maximum allowed alpha=4. s151 passes those provisional
   absolute thresholds, but H does not improve over B2 there.
3. **Anchor lift: fail.** H does not reduce alpha=4 misses by 50% or improve
   complete-hit by five percentage points on either scale.
4. **Runtime authorization: denied.** No HCA-driven residency or prefetch
   prototype, confirmatory five-seed expansion, or systems matrix is justified.

This kills **HCA as a directory** for the current research path. It does not
kill HCA's architectural value, nor the possibility that official V4 weights
behave differently.

## Next research path

The strongest remaining question is now narrower:

> Can B2 uncertainty identify the small set of prior-CSA anchor failures on
> which a compressed second-order block descriptor is worth reading?

The next pilot should compare B2, B2 plus a fixed COBS-style descriptor, and a
selective descriptor-acquisition gate at matched descriptor bytes. HCA stays
as an ablation only. Causal block-drop output damage must replace overlap as
the terminal label before any runtime work.

## Artifact bindings

Large checkpoints and raw derived rows remain ignored local artifacts.

| Artifact | SHA-256 |
| --- | --- |
| continuation pilot driver | `2c2d2f517dd279810a71370cc751e11b6d719d9646727bfc4526bf0e14786788` |
| original conditional analysis driver | `0b19975e57fbd360742dd14069f775e7b603bf470d211d3104bdcd82687cbea6` |
| clean-core sidecar reproduction driver | `6a6e627e688f44182f24611ab38c53d89d26110662a0865716f7c8aef8bb746d` |
| research sidecar source | `f0eae9119ff1d6ba29dd73ab86d7c2783621b236d01445726ec7e2deaeaf0681` |
| paired evaluation driver | `086ef65862745e4c6335981d7977ded1c82244f221d39852900f5f5547d77c4c` |
| s55 continuation summary | `93ac0d59b54a8264d5e4dad47aa750b1fe2b82c82de909e4ffbfe44cf5ebb362` |
| s151 continuation summary | `dc367e8508be784c1f41d3c947dee98a898725288cf05cfba32dfcac7039f938` |
| s55 stratified H0 summary | `fae1c54b46084afe693a3944508a1a346666669dc44c0cd7d2cf348f7b649b46` |
| s151 stratified H0 summary | `0bc128eadffa9de9350819ee2c6e1480e2c76a91d4833dd12b7999497e72a047` |
| s55 paired evaluation | `be6dc5d30e483603aa5186a70d98ceccdbeea094a6307b4f81ce79403941763e` |
| s151 paired evaluation | `77764cc59b35f3a81c4b04ec472ffe0fcdd3933e815d0d218f9d96d6b8ef9e7b` |

The initial diagnostic was run from source commit
`339670fd3ce868646e2153f090ccca05674c9662` plus uncommitted core
instrumentation. That instrumentation was removed. Both scales were then
re-extracted from the same checkpoint and generator state using the
clean-package sidecar above: the s55 row digest remained
`05c02d1da4837f19f8c51b69b58712f9b45e1b2038c4401bfc994719a1813f06`
and the s151 row digest remained
`29b93e34a0f8e18d714b3981e1c40dfb53ae7a30b202257657b2137613151828`;
both summary files remained byte-identical. The commit that contains this
report and sidecar is the authoritative reproducible code state. The pilot
remains non-confirmatory because it has only one continuation seed per scale.
