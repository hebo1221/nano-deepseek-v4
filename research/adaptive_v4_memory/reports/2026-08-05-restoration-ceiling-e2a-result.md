# Restoration Ceiling E2-A result

Date: 2026-08-05

Decision: **NO-GO — do not train a compact module to imitate full-compressed
behavior on this synthetic model. Pivot the primary branch to external
retrieval; retain K=8 only as a bounded diagnostic lead.**

E2-A completed the preregistered 160-cell no-training restoration-ceiling
audit. None of the five conjunctive gates passed. Full compressed memory had a
positive pooled mean on s151, but one checkpoint seed was negative at each
context, checkpoint-bootstrap intervals crossed zero, it did not consistently
outperform an eight-block native index route, and it was unsafe on the s55
saturation control.

This result rejects full-compressed behavior as a stable self-distillation
teacher for the planned synthetic restoration branch. It does not reject
restoration on natural models, external semantic retrieval, or a larger fixed
K as an engineering baseline.

## Frozen protocol

- Preregistration:
  [E2-A preregistration](2026-08-05-restoration-ceiling-e2a-preregistration.md)
- Frozen manifest:
  [`restoration-ceiling-e2a-v1.json`](../manifests/restoration-ceiling-e2a-v1.json)
- Experiment ID: `restoration-ceiling-e2a-v1`
- Implementation commit: `7311dc5b044ef3ccf4028af054d88c253cf9111b`
- Manifest-only commit: `dd05b76`
- Manifest SHA-256:
  `311c03fe6dcc962657febc60499d23b2b1c4d377b56af56f37f13e74b6c4972a`

No coordinate, seed, arm, metric, gate, or threshold changed after quality
access.

## Execution and integrity

The RTX 4090 service ran from 2026-08-05 16:04:54 to 16:05:56 KST and
completed in 62 seconds with zero restarts. All three quality-blind preflights
passed: s151/640 single retrieval, s151/1024 long generation, and s55/640
single retrieval.

An initial preflight invocation exited at import time because the remote shell
did not export the repository root on `PYTHONPATH`. It reached neither model
execution nor route construction and published no artifact. Repeating the
frozen command with the repository-standard `PYTHONPATH=.` resolved the
environment issue; no code or manifest amendment was required.

| Receipt | Result |
| --- | --- |
| Closed-world cells | 160 / 160 |
| Native vs identity replay | bitwise equal in 160 / 160 cells |
| Native vs equivalent absolute-K replay | bitwise equal in 160 / 160 cells |
| Shared workload digests | equal across checkpoint seeds and scales |
| Temporary files | 0 |
| Raw files | 160 |
| Raw byte count | 7,555,664 |
| Raw inventory SHA-256 | `7d0442a8cf6cafe25d71530dbb8fdbafaff5b7134be59f5bc6a588d9591afebe` |
| Remote/local summary SHA-256 | `a1ec317bdad70505eeb655a3c3e728e3c1b1b3a678fcf8cf47ceaa9570b65964` |
| Bootstrap | 200,000 deterministic checkpoint-cluster draws |

The remote summary and independently regenerated local summary are byte
identical. Raw artifacts remain intentionally untracked under
`artifacts/adaptive_v4_memory/restoration_ceiling_e2a/v1`.

Final repository validation after the result was regenerated locally:

- focused E2-A and manifest tests: 8 passed;
- full suite: 1,627 passed, 5 warnings in 306.07 seconds;
- Ruff: passed; and
- canonical package plus E2-A mypy scope: passed.

## Preregistered gate decision

| Gate | Frozen requirement | Observed result | Pass |
| --- | --- | --- | --- |
| s151/640 full teacher | positive in 5/5 seeds; 95% lower bound > 0 | positive in **4/5**; 95% CI **[-0.251, +2.014]** | No |
| s151/1024 length shift | positive in >=4/5 seeds and >=3/4 families; 95% lower bound > 0 | **4/5**, **4/4**; 95% CI **[-0.065, +1.150]** | No |
| Distributed beyond K=8 | full minus K8 positive in >=4/5 seeds; 95% lower bound > 0 | positive in **2/5**; 95% CI **[-0.522, +0.452]** | No |
| Evidence-headroom recovery | mean >=50%; 95% lower bound >=25% | mean **42.47%**; 95% CI **[-15.56%, 74.24%]** | No |
| s55 2x safety | lower bounds >-0.10 log-prob and >-2 pp accuracy | log-prob **-0.580**; accuracy **-10.0 pp** | No |

The frozen decision is
`NO_GO_FULL_CACHE_TEACHER_PIVOT_EXTERNAL_RETRIEVAL`.

## Primary details

### Full compressed memory versus native

| Context | Seed 6071406 | 6071407 | 6071408 | 6071409 | 6071410 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s151/640 | +0.711 | +2.890 | +0.739 | +0.664 | **-0.902** | +0.820 |
| s151/1024 | +0.411 | +1.642 | **-0.445** | +0.663 | +0.189 | +0.492 |

All four 1024 family means were positive: adversarial lexical distractors
+0.743, single remote retrieval +0.450, multiple independent needles +0.394,
and long generation +0.381. This explains the positive pooled mean, but the
checkpoint disagreement prevents a shared-teacher claim.

At s55/640, full minus native target log probability by seed was -0.268,
+0.132, +0.116, +0.823, and -1.044. Accuracy changes were 0, +3.125, -12.5,
+18.75, and -12.5 percentage points. The failure is therefore not only a weak
confidence interval; two checkpoints show material harm.

### Full compressed memory versus K=8

Across both s151 contexts, full minus K8 by seed was +0.438, +0.733, -0.714,
-0.084, and -0.548. Full attention was better in only two checkpoints.

At s151/640, K8 had a larger pooled gain than full memory (+0.931 versus
+0.820). At 1024, full was only slightly larger (+0.492 versus +0.451). The
context/family differences were mixed: full minus K8 ranged from +0.558 for
1024 adversarial distractors to -0.348 for 1024 long generation. This is not
evidence for a stable distributed-information advantage.

### Evidence anchor

Force-evidence minus native remained positive in all five checkpoint seeds at
both s151 contexts. At 640 the gains were +1.443, +3.432, +2.111, +1.159, and
+1.509; at 1024 they were +1.059, +2.292, +1.552, +1.291, and +1.173.

The project therefore still has a retrieval target: relevant evidence helps
reliably. What failed is using indiscriminate full compressed attention as the
teacher for recovering it.

## Prespecified capacity curve

Mean target-log-probability deltas from the native cohort budget were:

| Arm | s151/640 | s151/1024 | s55/640 2x |
| --- | ---: | ---: | ---: |
| index K=1 | 0.000 | 0.000 | -0.507 |
| index K=2 | +0.116 | +0.185 | -0.459 |
| index K=4 | +0.535 | +0.389 | 0.000 |
| index K=8 | **+0.931** | +0.451 | -0.066 |
| full compressed | +0.820 | **+0.492** | -0.048 |

This curve argues against interpreting E1 as “all extra memory is noise.” More
hard-route capacity often helps s151. It also argues against full memory as the
unique solution: K8 already matches or exceeds it in much of the panel while
reading far fewer blocks.

## Post-hoc diagnostic: the K=8 lead

This section is exploratory and cannot change the frozen decision.

- At s151/1024, K8 minus native was positive in all five checkpoint seeds and
  all four family means.
- At s151/640, it was positive in four of five seeds and all four family means;
  seed 6071406 remained negative at -0.111.
- Only 44/60 individual 640 cells and 40/60 individual 1024 cells were
  positive, so the aggregate pattern is not universal.
- K8 was not safe on s55: its checkpoint accuracy changes included -12.5
  percentage points.

K8 is therefore a useful baseline for a future retrieval experiment, not a
passed adaptive-memory result. A new fixed-K study should not be launched merely
to rescue the synthetic claim; it would need an independent natural-language
question and a physical memory/latency comparison.

## Interpretation and next decision

The H0/E1/E2 sequence now localizes the problem more sharply:

1. task evidence is consistently useful when identified;
2. the single best causal block is checkpoint-unstable;
3. an answer-free perturbation proxy does not recover it reliably; and
4. exposing every compressed block is also checkpoint-unstable and does not
   consistently beat K8.

Accordingly, do not train an IndexMem/RestoreKV-style module against this tiny
model's full-compressed outputs. Those methods remain relevant references—
[IndexMem](https://arxiv.org/abs/2605.25475) uses an explicit gated residual
for evicted information, while [RestoreKV](https://arxiv.org/abs/2608.01247)
self-distills a budget-matched restore cache—but E2-A shows that their teacher
assumption is not satisfied consistently here.

The primary research branch should move to **model-agnostic external semantic
retrieval on natural-language evidence**, using native K8 and full memory as
fixed baselines. The smallest defensible next experiment is not another large
synthetic matrix: it is a separately preregistered, natural-model pilot that
checks whether a frozen embedding/BM25-style retriever improves evidence recall
and task accuracy across multiple queries without checkpoint-specific causal
labels.

## Bottom line

The experiment saved the project from training a restoration mechanism toward
an unstable teacher. The remaining positive signal is narrower but more
actionable: evidence retrieval still has strong headroom, and moderate K can
help, but “read everything” is neither stable nor necessary. The next value is
in semantic retrieval and early natural-language validation, not more
synthetic selector or restoration training.
