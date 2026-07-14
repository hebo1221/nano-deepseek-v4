# P1 token-sized quality-path equivalence — S151

Date: 2026-07-14  
Decision: **chunk=1 accepted for S151 quality; chunk=2 rejected**

The S151 core policy set was compared against its frozen physical-tier,
token-by-token pilot reference. Chunk=2 produced a prediction mismatch on
`dense-global-aggregation` at context 512 and was rejected.

With chunk size 1 and the physical tier disabled, all 1,260 core-policy
conversation records and all 3,920 predictions matched the physical-tier
reference exactly. The summed forward time was 1,665.39 seconds on the recorded
host, about 4.4 times the S55 chunk=2 validation time.

The large quality matrix therefore uses a scale-specific execution contract:
S55 uses chunk=2, while S151 uses chunk=1. Both remain quality-only paths;
physical transfer, latency, and fused-runtime claims are evaluated separately.

The raw S151 equivalence SHA-256 is
`8981d93b2186c35a643a71d97731ee8bbc524f46f4c236780c084555d0ddb4dd`.
The checked result is `results/p1-token-cache-equivalence-s151.summary.json`.
