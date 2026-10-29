# Reuse accepted immutable view payloads — 2026-10-29

Existing 26.1/AR4/AR7.5 and 10.4/32.2 performance boundary. Follows the complete
scoped-batch correctness repair `9b364219`; no new checkbox or broader closure.

## Completed

- Source checks ask the canonical current-read cache for an immutable payload.
  Matching immutable payload identity replaces repeated full materialization and
  comparison. History/branch/head/clean-source checks still run first. Distinct
  payloads after invalidation receive canonical exact value/topology comparison.
- Output snapshot copying moves after batch preparation/schema checks and selected
  scope checks, so rejected input does not build an unused full output snapshot.
- Successful Graph and DurableGraph publication use one result helper to seed the
  existing weak current-read cache with the fully prebuilt accepted payload.
  Durable adoption occurs after append and successful graph execution; failures
  never install a projected but unaccepted state.
- A subsequent current capture shares the complete payload, even if the returned
  view selects only a few IDs. It still receives distinct capture identity,
  current branch/write provenance and the current retained-group catalog.
- Existing ordinary mutations and trusted pool access invalidate the cache.
  Same-head physical maintenance remains compatible through exact fallback.
  No new owner, permanent history pin, snapshot representation or retention policy.

## Verification

New native coverage checks complete-vs-selected membership on shared data,
immutable old inputs, ordinary mutation invalidation, exact same-head maintenance
fallback, metadata/catalog rebinding, durable adoption/reopen/foreign-owner
rejection, weak lifetime and distinct capture identity. The existing ten internal
pinned-write boundary cases, including raw untracked/corrupt-index rejection,
continue to pass. Checked operations still publish only once.

Final gates: **1,492 native / 54 ignored**, strict all-target core Clippy,
**2,511 root Python / 1 skipped**, **244 executable documentation examples**,
rebuilt/atomically staged full-ABI extension, scoped formatting/syntax/diff checks.
Three unrelated existing pipeline FFI warnings remain.

## Matched release evidence

Same `view_batch_publication` source and release profile as correctness baseline
`9b364219`. Both immutable sources stay retained as specified by that fixture:
a chain, f64 scalar and 16-element f64 vector per node, two selected nodes/one
selected edge, five sequential score updates with the original capture retained.
Three alternating quiet standalone process pairs give 15 publication samples per
size/variant. Output checks confirm state advancement, exact selected membership,
new scores and the unchanged original capture.

| Owner nodes | Baseline median | Reuse median | Baseline/reuse |
| --- | ---: | ---: | ---: |
| 1,000 | 1.464 ms | 0.517 ms | 2.83× |
| 100,000 | 310.443 ms | 76.283 ms | 4.07× |

Raw receipt: `../benchmarks/view-write-payload-reuse-2026-10-29.json`.
One additional macOS `/usr/bin/time -l` process per variant reports whole-fixture
peak resident memory of 594.281 MiB baseline and 589.359 MiB reuse. This includes
construction and both sizes, not isolated publication allocations; no meaningful
peak-memory reduction is claimed from those figures.

## Explicit remaining limits

An accepted result still clones the full GraphSnapshot and builds its immutable
existence/adjacency indexes. Scoped publication is therefore still linear in
owner payload/topology, not O(1) in changed cells. This optimization removes
redundant input work; it does not certify production memory/throughput, fsync,
Python GIL latency or all historical-capture workloads. Cold capture and full
output ownership remain in existing 10.4/32.2; default/filter ownership and general
compound topology execution retain their original boundaries.
