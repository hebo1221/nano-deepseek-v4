# P2 batch-size validation

## Decision

Batch 20 is rejected for the P2 Tier-S quality matrix. The frozen quality
protocol remains batch 4 for both S55 and S151.

## Counterexample

The S55 chunk-2 cache-faithful path compared batch 20 against five batch-4
slices over identical input tensors and policy settings. After 180 identical
policy-conversation records, `fixed-2x` produced the first disagreement on
`single-remote-retrieval`, context 128, row 0: batch 20 predicted token 112,
whereas batch 4 predicted token 108.

The attempted larger batch was faster in this prefix of the validation
(4,368.15 ms versus 8,960.23 ms summed policy wall time), but speed does not
override the preregistered exact-prediction contract. This result is evidence
of a batch-shape-sensitive BF16 execution path, not evidence for or against a
memory policy. The P2 runner now rejects any batch size other than 4 and audits
the batch size of every resumed shard.

## Provenance

- Raw artifact: `artifacts/adaptive_v4_memory/paper_grade/validation/p2-batch20-equivalence-s55.json`
- Raw SHA-256: `02ef7f1f477bb737c49ad2fa371fd065955589d6879bb85283e961e56fb32290`
- Checked summary: `research/adaptive_v4_memory/results/p2-batch20-equivalence-s55.summary.json`
- Validation source: `ecdbfda50a5973f5b6a8b9d4150faa0458e42318` (clean)
