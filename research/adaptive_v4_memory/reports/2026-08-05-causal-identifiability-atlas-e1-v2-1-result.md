# Causal Identifiability Atlas E1 v2.1 result

Date: 2026-08-05

Decision: **NO-GO — causal topology is unstable; do not train a shared hard
selector on these labels.**

This report closes the preregistered 160-cell, no-training identifiability
audit. Only one of five conjunctive primary gates passed. The result preserves
the earlier H0 finding that known task evidence creates substantial headroom,
but shows that the best equal-budget block is not stable across independently
trained checkpoints and is not recovered reliably by the frozen observable
proxy.

The result supports a narrow conclusion: the planned shared hard-routing
learner is not justified by this synthetic checkpoint cohort. It does not show
that adaptive memory is impossible, that task evidence is useless, or that a
soft integration/restoration method would fail.

## Frozen protocol

- Preregistration:
  [E1 preregistration](2026-08-05-causal-identifiability-atlas-e1-preregistration.md)
- Runtime correction:
  [v2 amendment](2026-08-05-causal-identifiability-atlas-e1-v2-runtime-amendment.md)
- Preflight correction:
  [v2.1 amendment](2026-08-05-causal-identifiability-atlas-e1-v2-1-preflight-amendment.md)
- Frozen manifest:
  [`causal-identifiability-atlas-e1-v2-1.json`](../manifests/causal-identifiability-atlas-e1-v2-1.json)
- Experiment ID: `causal-identifiability-atlas-e1-v2-1`
- Implementation source commit: `9e7a2ea5a854e0ff2738929539d07cb92324290a`
- Manifest-only commit: `afffda4`
- Manifest SHA-256:
  `a7347e10422ae7b6aba1ae5b41c48babc34365ded18a454edeb90c3bbf052854`

No primary gate, threshold, coordinate, seed, route cardinality, target, or
metric changed after quality access.

## Execution and integrity

The fresh v2.1 namespace ran on the RTX 4090 from 2026-08-05 12:50:05 to
13:38:37 KST, completing in 48 minutes 32 seconds. The service completed
normally with zero restarts.

| Receipt | Result |
| --- | --- |
| Closed-world cells | 160 / 160 |
| Exhaustive-atlas cells | 60 / 60 |
| Native vs identity-replay logits | bitwise equal in 160 / 160 cells |
| Shared workload input digests | equal across all five checkpoint seeds |
| Temporary files | 0 |
| Raw files | 160 |
| Raw byte count | 74,316,004 |
| Raw inventory SHA-256 | `5fb93034dbf39e6765059b74e649fdce1b1f366e6f638849a660d9524d8695a4` |
| Remote/local frozen-summary SHA-256 | `1fcd90296526b28f76cfd69cebd33a8ecf7346c4a0fd77670b4596862df645de` |
| Bootstrap | 200,000 deterministic checkpoint-cluster draws |

The final raw artifacts are under
`artifacts/adaptive_v4_memory/causal_identifiability_atlas_e1/v2-1`, and the
independently regenerated local summary is
`artifacts/adaptive_v4_memory/causal_identifiability_atlas_e1/v2-1.summary.local.json`.
Artifacts remain intentionally untracked.

The v1 run published 65 cells before a batch-shape-dependent secondary-teacher
invariant aborted coordinate 66. Those cells were quarantined and not reused.
V2 fixed final-chunk padding, but a targeted quality-blind preflight exposed a
row-count mismatch and published zero cells. V2.1 fixed route-plan construction,
passed both the first coordinate and the exact formerly failing coordinate,
then started in a new namespace without reusing either earlier run.

Final repository validation after the v2.1 fix:

- focused E1 tests: 8 passed;
- Ruff: passed;
- mypy: passed; and
- full suite: 1,621 passed, 5 warnings in 302.63 seconds.

## Preregistered gate decision

| Gate | Frozen requirement | Observed result | Pass |
| --- | --- | --- | --- |
| Causal topology | mean pairwise singleton Jaccard >= 0.60 | **0.3217**, 600 pairwise comparisons over 60 strata | No |
| Observable alignment | mean proxy/utility Spearman >= 0.30 in all 5 seeds | **-0.043, -0.026, 0.181, 0.155, 0.156** | No |
| s151/640 recovery | positive in 5/5 seeds; recovery >= 25%; 95% lower bound >= 10% | positive in **2/5**; recovery **-4.76%**; 95% CI **[-22.49%, 14.07%]** | No |
| s151/1024 length shift | positive in >= 4/5 seeds and >= 3/4 families | positive in **4/5** seeds and **3/4** families | Yes |
| s55 2x safety | 95% lower bounds > -0.10 log-prob and > -2 pp accuracy | log-prob **+0.053**; accuracy **-2.5 pp** | No |

Therefore `all_primary_gates_pass=false`, with frozen decision
`UNSTABLE_CAUSAL_TOPOLOGY_DO_NOT_TRAIN_SHARED_SELECTOR`.

### End-task details

At s151/640, perturbation-proxy minus native target log probability by seed
was:

| Checkpoint seed | Delta |
| ---: | ---: |
| 6071406 | -0.3864 |
| 6071407 | -0.1668 |
| 6071408 | +0.3757 |
| 6071409 | -0.5343 |
| 6071410 | +0.2986 |

The matched evidence route was positive in all five seeds: +1.3266, +2.7799,
+1.7570, +1.7475, and +1.0700. The H0 headroom is therefore present in E1,
but the observable proxy does not recover it.

At s151/1024, proxy minus native was positive in four seeds:
+0.0010, +0.4243, +0.3762, -0.1598, and +0.6811. Family means were:

| Family | Target log-prob delta |
| --- | ---: |
| single remote retrieval | +0.7350 |
| multiple independent needles | +0.3297 |
| adversarial lexical distractors | +0.2777 |
| long generation changing evidence | -0.2842 |

This is a useful positive diagnostic, but it cannot override the conjunctive
decision. A post-hoc checkpoint-cluster interval for the pooled 1024 effect was
[-0.0108, +0.5174], and both one checkpoint and the long-generation family
remained negative.

At s55/640 with the 2x budget, proxy minus native target log probability had a
95% interval of [+0.0534, +0.5910]. Accuracy had a 95% interval of
[-2.5, +12.5] percentage points, narrowly missing the prespecified lower-bound
safety threshold of greater than -2 points.

## Prespecified and post-hoc diagnosis

The following diagnostics do not replace the failed primary gates.

1. **Task evidence is helpful but not a unique best block.** Its equal-budget
   utility was positive in 86% of exhaustive checkpoint/workload/layer strata,
   yet it was the highest-utility block in only 55.7% and the most necessary
   block in 50.0%. The synthetic evidence annotation is therefore a useful
   intervention, not a reliable universal winner label.
2. **Instability is not explained only by near-ties.** The median gap between
   the first- and second-ranked utility blocks was 0.1247; its 75th and 90th
   percentiles were 0.4998 and 1.4015. Expanding singleton winners to top-2,
   top-4, or top-8 sets reduced cross-checkpoint Jaccard further to 0.1761,
   0.1277, and 0.0908.
3. **The proxy is a refined full-read heuristic, not a causal identifier.** Its
   route Jaccard with full-read scoring was 0.781--0.854, versus only
   0.419--0.488 with the native route. Its top-m recall of the true utility
   winner remained about 0.36--0.39 for m in {1, 2, 4, 8}.
4. **A self-teacher target does not rescue shared labels.** Teacher-token causal
   topology stability was 0.1467, below gold-target stability of 0.3217, and
   gold/teacher top-1 agreement was 0.37.
5. **The failure is conditional, not uniformly negative.** Proxy routing helped
   several 1024-context families and the s55 aggregate, but harmed s151/640 and
   long-generation shift. A single global winner-take-all rule conflates these
   regimes.

These observations are consistent with redundancy and nonlinear interactions:
multiple blocks can substitute for one another, and the marginal value of a
block depends on the checkpoint and retained set. An additive `K=1` winner
label is consequently too brittle to supervise one shared hard selector.

## Research decision

Do not launch the previously contemplated 1,920-workload shared reranker. More
synthetic labels from the same target definition would scale the unstable
supervision rather than resolve it.

The next experiment, if separately approved and preregistered, should change
the intervention rather than enlarge the learner:

1. test a small **query-conditioned residual/soft integration** mechanism that
   restores information without naming one winning block; or
2. test a **model-agnostic external semantic index** whose target is retrieval
   relevance rather than checkpoint-specific marginal log-probability.

A bounded E2 should first use the same five-checkpoint design to establish
cross-checkpoint stability, then move immediately to a small natural-language
transfer panel. It should retain native/full-read controls, report physical
HBM and latency, and stop before training if the new representation again
fails a no-training identifiability gate.

This direction follows the distinction between output-perturbation scoring and
causal evidence sets represented by [CriticalKV-style scoring
(arXiv:2502.03805)](https://arxiv.org/abs/2502.03805), [causal evidence
analysis (arXiv:2607.21692)](https://arxiv.org/abs/2607.21692), and
[counterfactual routing (arXiv:2608.01676)](https://arxiv.org/abs/2608.01676).
Restoration alternatives to study include [IndexMem
(arXiv:2605.25475)](https://arxiv.org/abs/2605.25475), [ResKV
(arXiv:2607.29591)](https://arxiv.org/abs/2607.29591), and [RestoreKV
(arXiv:2608.01247)](https://arxiv.org/abs/2608.01247).

## Bottom line

H0 was not wasted: it showed that the model can benefit strongly when the
right evidence is retained. E1 supplied the missing falsification test and
located the bottleneck: under this hard-routing formulation, “the right block”
is neither stable enough across checkpoints nor observable enough from the
frozen proxy. The defensible move is to stop the shared-selector branch and
redirect compute toward integration/restoration plus early natural-language
validation.
