# Legacy M3 offline learned risk pilot — negative result

Status: M3 failed its positive gate at both Tier-S scales. M4 will retain the
M2 training-free controller.

## Protocol

Request-level indexer features and quality-derived top-k/dense labels were split
by independent generated-example seeds: 768 train, 384 calibration, and 768
test examples per scale. A two-head MLP predicts sufficient top-k with a 4x
under-allocation penalty and dense-memory risk with binary cross entropy.
Budget offset and fallback threshold were fixed on calibration data before the
test split.

This pilot is offline: it first collects the final-query native probe from a
complete model pass, derives the learned allocation, and replays a selection
plan. It therefore does not implement or evaluate a deployable token-t to
token-(t+1) learned lookahead controller.

The checked [`summary`](../results/m3-tier-s-learned-risk-controller.summary.json)
references ignored raw results by SHA-256. No tokens or model weights are
committed.

## Results

At S55, native features reduced budget MAE from the context+layer ablation's
11.67 to 6.39, but under-allocation worsened from 5.5% to 9.9%. The calibrated
dense threshold fell back on 63.0% of test examples. Learned allocation reached
72.0% accuracy using 36.6 blocks, while M2 reached 66.1% with 6 blocks; this is
not a Pareto improvement.

At S151, MAE improved only from 5.87 to 5.32 and under-allocation changed from
10.29% to 10.42%. Fallback ran on 57.0% of examples. Learned allocation reached
75.8% with 54.8 blocks versus M2 at 73.6% with 5 blocks, again failing Pareto.

The symmetric-loss ablation had worse coverage/MAE at both scales. An 8-hidden
model reduced S151 fallback but did not change the decision. A refresh ablation
was impossible on this single-answer-control-point dataset, independently
blocking a positive M3 claim.

## Decision

The offline learned model detects hard cases but its calibrated fallback is too broad;
the modest quality gain is purchased with roughly 6–11x more selected blocks.
The simpler M2 rule remains the supported research path. M4 may proceed with
that controller, and this M3 implementation remains as a reproducible offline
negative baseline rather than being tuned on the test set. The separate online
learned-lookahead question remains open under the paper-grade protocol.
