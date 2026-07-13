# M1 Tier-T replay smoke report

Status: correctness evidence only; not eligible for a model-quality or systems claim.

## Scope

The deterministic run used nine untrained tiny FP32 CPU models: three seeds at
16, 24, and 32 tokens, with two CSA layers. The three 16-token traces were used
for static calibration and the other six traces for replay evaluation. No raw
tokens or prompts were persisted.

The checked result is
[`m1-tier-t-smoke.summary.json`](../results/m1-tier-t-smoke.summary.json),
produced by [`run_m1_tier_t_smoke.py`](../scripts/run_m1_tier_t_smoke.py) from
the immutable [`manifest`](../manifests/m1-tier-t-smoke.json).

## Correctness observations

- All seven policies replayed 336 CSA queries deterministically.
- The fixed top-k, recency, random, and index-reuse policies each selected 552
  blocks (61,824 bytes), providing a memory-matched comparison.
- Fixed top-k-2 recovered 72.0% of native selections; recency-2 recovered 68.0%
  and random-2 recovered 67.5%. These are selection-recall diagnostics, not
  downstream quality scores.
- Top-p 0.9 recovered every native-selected block but used 1,116 blocks versus
  native's 912, so it is not memory matched.
- Exhaustive enumeration recovered the four-block native proxy set in all six
  final-query checks. This validates search mechanics only.
- The native-budget proxy regression reduced grouped holdout MAE from 0.5000 to
  0.1709 when score/overlap features were added. Because the target is native
  top-k count rather than a quality-derived sufficient budget, this result does
  not pass the preregistered M1 predictive gate.

## Gate decision

M1 infrastructure and Tier-T correctness are working, but the predictive gate
remains **not evaluated**. The next required artifact is a trained Tier-S model
with non-trivial long-context behavior and counterfactual quality-derived
sufficient-budget labels. M2 controller work must not use this smoke result as
evidence that adaptive allocation is safe.
