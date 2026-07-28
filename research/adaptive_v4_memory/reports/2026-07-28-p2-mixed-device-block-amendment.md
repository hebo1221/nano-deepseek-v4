# P2 mixed-device block amendment

This amendment changes execution topology only. The frozen 9,000-cell grid, 17 arms,
within-cell arm rotation, examples, seeds, estimands, and success gates do not change.
The existing final-3 namespace remains immutable bridge evidence and is not merged into
either new primary site ledger.

The primary grid is blocked by physical device. Every complete
`(scale, training_seed, budget, family, context)` stratum contains ten replicates.
Let the zero-based factor indices be `s`, `t`, `b`, `f`, and `c`. Define

`offset = (s + t + 5*b + f + 2*c) mod 10`.

GB10 receives replicate indices `offset + {0,1,2,3} mod 10`; RTX 4090 receives the
complement. This assigns exactly 3,600 and 5,400 cells. It is quality-blind and gives
each replicate index exactly equal frequency within every level of scale, training
seed, budget, family, and context. Each site then partitions its canonical subsequence
across three local workers by site-local index modulo three.

Device is a preregistered block and a heterogeneity factor, not an exchangeable worker.
Primary arm contrasts remain within-cell. The analysis must report device-by-arm
interactions and site-stratified effects before a pooled effect. A pooled effect is
allowed only with the device term retained. The short synthetic phase-A trace mismatch
between GB10 and RTX 4090 is disclosed; the phase-B digest matched exactly.

Before full resume, each site must pass prerequisites-only and a fresh one-cell
schema, HMAC, selected-device, closed-world, and integrity gate. The first cell on each
site already exists in final-3 on GB10, so those two cells form a quality-blind bridge.
Bridge comparison may inspect prediction/token digests and systems counters, but it
may not change the assignment, arm inventory, seeds, analysis, or continuation rule.

The GB10 site inherits the exact v1.3.4 execution projection. The RTX 4090 site freezes
its own Python, Torch, CUDA, kernel, architecture, GPU class, and routing identity in
its implementation and activation. Site output, activation, worker-ledger, and
persistent-session namespaces are disjoint. Final-3 is stopped only after both new
site prerequisites are ready; its last committed ledgers are snapshotted as bridge
lineage, and any in-flight uncommitted files are excluded.

The first RTX 4090 fresh gate in the unsuffixed mixed-site namespace committed zero
shards and materialized no prediction. It stopped before quality-evaluator launch
because the persistent-session planner still used the superseded global modulo
assignment. That namespace remains immutable. Retry-1 changes only the planner's
assignment source to the preregistered device block followed by site-local modulo;
the grid, arms, seeds, within-cell order, estimands, and success gates remain unchanged.
