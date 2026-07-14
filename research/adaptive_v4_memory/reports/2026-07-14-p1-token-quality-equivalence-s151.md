# P1 token-sized quality-path equivalence — S151

Date: 2026-07-14  
Decision: **chunk=1 accepted for S151 quality; chunk=2 rejected**

The S151 core policy set was compared against its frozen physical-tier,
token-by-token pilot reference. Chunk=2 produced a prediction mismatch on
`dense-global-aggregation` at context 512 and was rejected.

A clean-source chunk-size-1 rerun reproduced exact agreement for all 1,260
core-policy conversation records and all 3,920 predictions. The summed forward
time was 935.10 seconds on the recorded host.

The large quality matrix therefore uses a scale-specific execution contract:
S55 uses chunk=2 and S151 uses chunk=1. Both are quality-only paths; physical
transfer, latency, and fused-runtime claims are evaluated separately.

The accepted raw S151 equivalence SHA-256 is
`1722df136da0f4c963036ed12301af287c7b62c6f516e1874fe796faa44ee557`.
The checked result is `results/p1-token-cache-equivalence-s151.summary.json`.
The invalidated former digest remains absent from the evidence chain.
