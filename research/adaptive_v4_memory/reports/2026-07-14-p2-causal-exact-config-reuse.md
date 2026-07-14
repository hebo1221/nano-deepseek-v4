# P2 causal exact-config reuse amendment

## Timing and decision

This amendment was frozen before any held-out P2 causal-factorial shard was
generated. The 9,000-shard Cartesian design, 20 conversations per shard, 16 arm
labels, paired inputs, and batch size 4 remain fixed. Four arms receive direct
physical-path measurements: the central fixed/adaptive pair and both added
top-p baselines. The top-p contrasts are supplemental; the primary causal gate
and original component contrasts remain unchanged.

Some calibration cells produce multiple preregistered arm labels whose complete
`SameTokenControllerConfig` objects are identical. Running the same model,
input, and controller config repeatedly would add inference cost but no new
experimental unit. Within a single paired batch, the runner therefore executes
the first occurrence and reuses it only when:

1. the canonical JSON representation of the full config has the same SHA-256;
2. exact dataclass equality also holds; and
3. the source arm was executed earlier in that batch.

A digest collision or provenance mismatch fails the audit. Reused rows record
zero executed wall time, the source arm, and the config digest. Quality and
physical executed-forward counts are reported separately from arm-conversation
counts, so reuse cannot be presented as an independent execution.

## Claim boundary

This is an execution optimization for structurally identical conditions, not a
reduction in sample size and not an additional replication. Configurations that
differ by even one quota, signal flag, pin flag, refresh rule, fallback rule, or
schedule variant still execute independently. The final inference remains
clustered by the five training seeds and paired by conversation.
