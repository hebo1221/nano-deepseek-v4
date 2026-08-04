# Causal-evidence route H0 pilot result

Date: 2026-08-05

Decision: **the frozen H0 gate passes. Ground-truth evidence routing has a large,
reproducible oracle gap, including at the P2 2x fixed budget. A small
causal-supervision selector pilot is scientifically justified; a larger matrix,
deployability claim, or natural-language claim is not.**

This result follows the pre-outcome freeze in
`causal-evidence-route-h0-v1.json`. It is a 180-cell synthetic diagnostic, not a
new Adaptive V4 performance claim.

## Execution and integrity

- Source commit: `e81b452` (detached exact checkout on RTX4090).
- Frozen manifest SHA-256:
  `affc128d5764da9ef070a29ae90149f3a6fd5224adb5ae1b4d3621ee1bea5870`.
- Final run: 2026-08-05 04:20:09--04:20:42 KST; 180/180 cells in 33 seconds.
- Result inventory SHA-256:
  `26935b0832ecf13a80d770491afd645a7bc9bb420bcbcc88073af11334c771be`.
- Closed world: zero missing, unexpected, or temporary cells.
- Identity replay: 180/180 stored outcomes exactly match native. The runner also
  required bitwise-equal full logits before publishing each cell.
- Route contract: all 1,440 layer-query receipts have equal cardinality and
  distinct evidence/control anchors.

The first service invocation had no repository `PYTHONPATH` and failed at import
before producing a cell. It failed three times total (the initial attempt and
two automatic retries), still with zero cells. The unit was stopped and
relaunched with only the checkout path added; the final run had no retry or cell
rewrite. This startup correction does not change the frozen code, manifest,
coordinate seeds, checkpoints, or outcomes.

## Frozen gate result

The primary estimand is `force-evidence - matched-control`. Each frozen
coordinate is weighted equally; multiple queries inside a coordinate are first
averaged. The pooled interval uses 200,000 paired-coordinate bootstrap draws
with seed 20260805.

| Budget / scale | Target log-prob delta | Accuracy delta | Adversarial poison-margin delta |
| --- | ---: | ---: | ---: |
| 1x / s55 | **+3.407** | **+60.0 pp** | **-5.338** |
| 1x / s151 | **+3.656** | **+76.7 pp** | **-5.807** |
| 2x / s55 | **+3.179** | **+45.6 pp** | **-2.736** |
| 2x / s151 | **+3.112** | **+52.8 pp** | **-4.644** |

At 1x, all six scale-by-training-seed strata are positive. The pooled mean is
`+3.531`, with coordinate-bootstrap 95% interval `[+3.162, +3.899]`. A disclosed
scale-seed cluster bootstrap, consistent with checkpoints as independent units,
is also positive: `[+2.787, +4.292]`. Accuracy is positive on both scales, and
the wrong-minus-gold margin decreases on both scales. The predeclared GO gate
therefore passes every clause.

The 2x target-log-probability effect remains positive on both scales, so there
is no sign reversal at the exact P2 fixed-budget comparator. Query-weighted
calculations preserve every sign; the result is not caused by the four-query
long-generation family receiving less weight in the primary family-balanced
analysis.

## The decision-relevant headroom

`force-evidence - matched-control` establishes that the selected evidence block
causally matters, but matched control is deliberately information-poor. The
more practical oracle-headroom contrast is `force-evidence - native`:

| Budget / scale | Target log-prob delta | Accuracy delta |
| --- | ---: | ---: |
| 1x / s55 | **+0.894** | **+16.7 pp** |
| 1x / s151 | **+2.274** | **+55.6 pp** |
| 2x / s55 | **+0.409** | **+2.2 pp** |
| 2x / s151 | **+1.306** | **+28.3 pp** |

This contrast is positive in all six scale-seed strata at 1x. At 2x it is
positive in five strata and exactly zero for s55 seed 6071406. Native routes
contain the evidence anchor in 68.1%/47.9% of layer-query receipts at 1x for
s55/s151, increasing to 82.6%/65.0% at 2x. The remaining route error is therefore
substantial for s151 and low-budget s55, while s55 at 2x is near saturation.

## Interpretation against current research

The result supports the route-first branch suggested by ICML 2026
[CriticalKV](https://arxiv.org/abs/2502.03805), recent
[causal evidence sets](https://arxiv.org/abs/2607.21692), and
[counterfactual route evaluation](https://arxiv.org/abs/2608.01676): supervise
selection with output-relevant counterfactual labels, not selector-score demand.
It also respects the warning from
[Error Certificates for KV-Cache Eviction](https://arxiv.org/abs/2607.21475):
cache-side state alone need not reveal what an eviction destroys.

The experiment does **not** show that the evidence block is inferable without a
ground-truth position, that a learned router will recover the oracle gap, or that
the gain transfers to natural text. It shows that routing error is causally
important enough to make those questions worth testing. If a compact selector
cannot recover a meaningful fraction of the `force-evidence - native` gap, the
next branch remains residual restoration/reconstruction, consistent with
[IndexMem](https://arxiv.org/abs/2605.25475),
[RestoreKV](https://arxiv.org/abs/2608.01247), and
[ResKV](https://arxiv.org/abs/2607.29591).

## Next decision

The next authorized scientific unit should be a small causal-supervision
selector pilot, concentrated on s151 and the 1x stress budget, with s55 2x as a
saturation control. It must measure recovered fraction of oracle headroom,
selector recall, end-task accuracy, and routing overhead. No second experiment
was launched here: the user approved one new bounded experiment, and this H0
pilot is that experiment.
