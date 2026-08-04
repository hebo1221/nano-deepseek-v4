# P2 lean-prefix postmortem and causal-route H0

Date: 2026-08-05

Decision: **retire score-demand calibrated layer quotas as the primary Adaptive V4
claim, retain protected evidence as a useful mechanism, and run one bounded
causal-route oracle pilot before choosing between a learned selector and residual
restoration.**

The 4,950-cell result is an outcome-informed futility prefix. It is not the
pre-registered 9,000-cell confirmatory result and cannot establish two-scale,
five-seed success. It is nevertheless enough to reject further large-matrix work
on the current quota rule: the observed direction is adverse, the pin effect is
cleanly separable, and the largest quota distortions produce the largest losses.

## Closure and claim boundary

The boundary closed at 2026-08-05 04:10:52 KST.

- GB10: 1,980 cells; runner inactive and disabled; stop guard active.
- RTX4090: 2,970 cells; runner inactive and disabled; stop guard active.
- Both final target cells exist and both first 4x future cells are absent.
- All 4,950 payloads passed their frozen schema, payload digest, coordinate,
  execution-order, and actual-hot-memory checks.
- The prefix contains all five s55 seeds at 2x and 4x, but only s151 seed
  6071406 at 2x. The formal primary gate is therefore not evaluable.

The independent unit is the training seed. Query counts below are descriptive;
they are not treated as independent replicates.

## Primary result

The estimand is `calibrated+pins - fixed+pins` paired answer accuracy.

| Scale / budget | Seeds | Mean delta | Seed bootstrap 95% interval | Seed directions |
| --- | ---: | ---: | ---: | ---: |
| s55 / 2x | 5 | **-1.026 pp** | [-2.090, +0.115] pp | 1 positive, 3 negative, 1 zero |
| s55 / 4x | 5 | **-0.308 pp** | **[-0.565, -0.025] pp** | 1 positive, 4 negative |
| s151 / 2x | 1 | **-0.193 pp** | not inferential | negative |

Each seed contributes 28,000 paired query outcomes. The s55 2x seed deltas were
`+0.671, -1.439, -1.361, 0.000, -3.004` pp; the 4x deltas were
`+0.157, -0.382, -0.468, -0.093, -0.754` pp. Absolute mean accuracies were:

| Scale / budget | fixed+pins | calibrated+pins |
| --- | ---: | ---: |
| s55 / 2x | 48.525% | 47.499% |
| s55 / 4x | 49.643% | 49.335% |
| s151 / 2x, one seed | 39.686% | 39.493% |

The result is not a near-miss for the intended claim. It is negative at both
observed scales, negative for four of five s55 seeds at 4x, and its only exactly
zero s55 2x seed is structurally uninformative rather than reassuring.

## Causal decomposition

The 2x2 arms separate quota and protected-pin effects.

| Estimand | s55 / 2x | s55 / 4x | s151 / 2x, one seed |
| --- | ---: | ---: | ---: |
| calibrated-no-pins - fixed | **-1.254 pp** | **-0.342 pp** | **-1.211 pp** |
| fixed+pins - fixed | **+0.744 pp** | **+0.354 pp** | **+1.132 pp** |
| calibrated+pins - calibrated-no-pins | **+0.971 pp** | **+0.388 pp** | **+2.150 pp** |

Both pin contrasts are positive in all five s55 seeds at both budgets. The quota
contrast is negative in the same pattern as the primary contrast. Pins partially
repair the calibrated policy, but they do not overcome its allocation loss.

The family split identifies the failure rather than averaging it away:

- s55 2x adversarial lexical distractors: -2.44 pp;
- s151 2x single remote retrieval: -5.50 pp;
- s151 2x adversarial lexical distractors: -4.50 pp;
- s151 2x instruction persistence: +6.25 pp.

At s55, every context from 80 through 1,024 is negative at both 2x and 4x.
The positive s151 instruction slice therefore shows specialization, not a general
quota advantage: protected/persistent content can benefit while retrieval and
competing evidence lose.

## Why the quota rule failed

1. **Its target is indirect.** Calibration used requested-block counts and
   candidate counts derived from selector scores; it did not use targets or
   output damage. The experiment shows that score demand is not a safe proxy for
   where a fixed global budget produces answer value.
2. **The strongest reallocations are the worst.** At s55 2x, seed 6071410 moves
   the balanced `4/4/4` budget to `8/2/2` and loses 3.004 pp. At 4x it moves
   `8/8/8` to `16/3/5` and loses 0.754 pp.
3. **One nominal seed is structurally null.** s55 seed 6071409 uses `4/4/4` for
   both calibrated and fixed at 2x, producing 28,000 exact ties. A strict
   all-five-seed positive gate was unattainable in that cell.
4. **The synthetic models are a stress laboratory.** They were trained at short
   lengths and evaluated through 1,024 tokens. The result can kill this controller
   rule inside the laboratory, but it is not evidence about official DeepSeek-V4
   weights or natural-language transfer.

Actual hot-block and hot-byte sequences were exactly matched by contract. Small
transfer and allocator differences remain descriptive implementation effects,
not a production systems comparison.

## Updated research map

The field has moved away from a single generic "adaptive cache" path:

| Path | Representative evidence | Implication here |
| --- | --- | --- |
| Outcome- or causality-aware selection | ICML 2026 [CriticalKV](https://arxiv.org/abs/2502.03805), [causal evidence sets](https://arxiv.org/abs/2607.21692), and August [counterfactual route replay](https://arxiv.org/abs/2608.01676) | Test an equal-budget causal oracle before training another router. |
| Conditional memory access | ICML 2026 [Learning When to Attend](https://arxiv.org/abs/2603.17484) and [uncertainty-gated selection](https://arxiv.org/abs/2607.07724) | Token/query gating is more plausible than a static per-layer quota, but only if an oracle gap exists. |
| Selection-cost reuse and hierarchy | ICML 2026 [Sketch-and-Walk](https://arxiv.org/abs/2602.07397), [ReTopK](https://arxiv.org/abs/2607.27692), and [LongCat Sparse Attention](https://arxiv.org/abs/2608.01662) | These optimize a selector that is already good; they do not repair the negative quality estimand. |
| Residual restoration instead of better eviction | ICML 2026 [IndexMem](https://arxiv.org/abs/2605.25475), [RestoreKV](https://arxiv.org/abs/2608.01247), and [ResKV](https://arxiv.org/abs/2607.29591) | This becomes the preferred branch if perfect evidence routing has little headroom. |
| Audit and attribution | [Error Certificates for KV-Cache Eviction](https://arxiv.org/abs/2607.21475) and [Does Accuracy Equal Evidence?](https://arxiv.org/abs/2608.01631) | Cache-side signals may not predict destroyed information, and accuracy alone can hide evidence loss. Record target log-probability and poison margin. |

This map rules out another large score-based quota sweep. It also makes a
systems-only optimization premature: first determine whether the remaining error
comes from selecting the wrong block or from losing/integrating its contents.

## One next experiment: causal-evidence route H0

The next pilot is frozen at 180 cells:

`2 scales x 3 independent model seeds x 2 budgets x 1 context x 3 families x 5 replicates`.

- scales: s55 and s151;
- model seeds: 6071406, 6071407, 6071408;
- context: 640;
- budgets: 1x stress budget and 2x, where 2x matches the P2 balanced fixed
  per-layer comparator;
- families: single remote retrieval, adversarial lexical distractors, and
  long-generation changing evidence;
- routes: native, exact identity replay, force ground-truth evidence, and a
  matched control that differs by exactly one block at every CSA layer/query.

The route cardinality is identical. Identity replay must be bitwise equal to the
native logits in every cell. Primary outcomes are target log-probability and
accuracy; the adversarial family additionally records the maximum
wrong-minus-gold log-probability margin.

The predeclared 1x GO gate is:

1. `force-evidence - matched-control` target log-probability is positive on both
   scales and in at least five of six scale-by-seed strata;
2. the coordinate-bootstrap pooled 95% interval excludes zero in the positive
   direction;
3. answer accuracy is non-negative on both scales; and
4. force-evidence reduces the poison margin on both scales.

The 2x arm measures whether the oracle gap persists at the exact P2 fixed budget.
A negative sign reversal fails the mechanism; saturation near zero is allowed but
does not justify a learned router by itself.

This is an outcome-aware upper bound, not a deployable policy. If it fails, no
router training, larger seed matrix, or new runtime is authorized; the next
research branch is residual restoration/reconstruction. If it passes, only then
is a small causal-supervision router pilot justified.

## Artifact and implementation bindings

| Artifact | Binding |
| --- | --- |
| closed-prefix summary | `18885bc011153c07b4dd99f27d42d5d0c6638a3a932d53aa53e2e5c075eb6a93` (131,723 bytes) |
| prefix analyzer | `dc9b46f08f09ca96ca38b81e00ae3ffdf41dd2ef5d7efae888f6c9a76c1cdb60` |
| causal-route pilot driver | `8e57e6ee710dd3135443edec78dc1f4493a46189a227c4b256d40b52536619ab` |
| implementation commit | `e5b493982075597b0d38f242af5ae3243459db6d` |

The implementation commit was pushed before any 4,950-cell outcome was opened.
It passed the focused tests, Ruff, mypy, and the repository's 1,613-test suite.
The execution manifest is a separate, post-analysis freeze and contains no code
change.
