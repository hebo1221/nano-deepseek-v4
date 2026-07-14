# P1 token-sized quality-path equivalence — S151

Date: 2026-07-14  
Decision: **chunk=2 rejected; chunk=1 revalidation pending**

The S151 core policy set was compared against its frozen physical-tier,
token-by-token pilot reference. Chunk=2 produced a prediction mismatch on
`dense-global-aggregation` at context 512 and was rejected.

An earlier chunk-size-1 run reported exact agreement for all 1,260 core-policy
conversation records and 3,920 predictions, but its digest-bound raw artifact
was subsequently overwritten by a dirty-source rerun. The checked summary has
therefore been invalidated until a clean-source rerun reproduces the result.

The large quality matrix must not execute S151 until the clean chunk-size-1
artifact is regenerated and audited. S55 remains approved at chunk=2. Both are
quality-only paths; physical transfer, latency, and fused-runtime claims are
evaluated separately.

The former raw digest
`8981d93b2186c35a643a71d97731ee8bbc524f46f4c236780c084555d0ddb4dd`
is retained only as invalidation history; it is not accepted evidence because
the corresponding raw bytes are no longer present.
