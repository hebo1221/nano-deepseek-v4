# P2 direct controller v1.3.3 reuse-admission view amendment

## Decision

Revision v1.3.2 is retired before activation and before any held-out quality
work.  Its manifest and signed admission/genesis pair remain immutable
historical evidence.  A distinct prospective revision, v1.3.3, corrects the
producer/consumer public-binding mismatch and uses new admission, activation,
output, worker, persistent-session, integrity, and summary namespaces.

## Observed boundary

The v1.3.2 static bundle was published and authenticated successfully.  The
first read-only arm construction then raised `ValueError: Reuse-admission
public binding schema drifted.`  The frozen contract consumer admitted exactly
eight fields, while the signed producer and genesis cross-binding contained
those eight plus:

- `superseded_failure_lineage_sha256`
- `superseded_failure_lineage_projection_sha256`

The v1.3.2 activation root, output root, worker ledger, persistent-session
ledger, integrity result, and summary are absent.  Its admission and genesis
state records zero completed shards, no quality RNG initialization, no input,
prediction, outcome, or aggregate materialization, and no active claim or
worker ledger.  The exception was not a signed terminal receipt; it is a
deterministic diagnosis reproduced from the frozen implementation and the
signed ten-field artifacts.

## Immutable anchors

- implementation commit: `92dec29cf6b9e4a4763a80d97e1ccd4569bfa5d6`
- result/manifest commit: `62f33e9809f76e31c3f7a216b7621d31469488a4`
- manifest SHA-256: `14c45f4efd2d0fdde9619944908fa22994524a36e22ec85e3fa596108f7f2056`
- admission SHA-256: `5ae8f4d7bc67752ff491db2781b2a8e5130bc6b26cc49648428da3aba886393d`
- genesis SHA-256: `05363b86f5a9cdba0097dfbdb8b9af7810a851c01d7bbe872e71fffa62e28beb`

The v1.3.3 historical loader must verify these Git and filesystem anchors,
the external-key HMACs, the exact-two closed world, all cross-links, the zero
quality state, and continued absence of every v1.3.2 prospective quality path.
It must not require the current index to equal the old implementation commit.

## Prospective amendment

The scientific grid, 9,000 coordinates, 17 arm order and semantics, seeds,
estimands, multiplicity rules, and confirmatory success gate do not change.
The contract now has one versioned ten-field admission-view schema shared by
producer, consumer, generator, and generated evaluator.  Both immediate
lineage fields are validated against the full lineage payload and its
normalized projection, not merely checked for SHA-shaped strings.

After the v1.3.3 implementation commit and manifest-only commit, a fresh
v1.3.3 exact-two admission/genesis pair may be published.  Activation remains
separate and immutable.  The required sequence is prerequisites-only, fresh
activation, zero-work ready-only preflight, authenticated zero matrix, one
operational cell with invariant-only checks, and outcome-independent full
resume.

## Threat boundary

Canonical v1.3.2 entrypoints are retired.  The new runner revalidates the old
static pair and old prospective-path absence before publication and quality
mutation.  An owner deliberately launching the frozen C4 code out of band
while holding the old attestation key remains outside the existing threat
model; defending that stronger case requires OS-level fencing or key rotation,
not reinterpretation of historical receipts.

Required reporting label:
`quality-blind-v1.3.3-reuse-admission-view-schema-amendment`.
