# Single-buffer recorded-state reconstruction

Existing 12.2/16.2/32.2 retained-state and performance scope. No new checklist rows.
**201 checked / 67 open**. This precedes the deeper 33.2 integrity scan.

## Completed

HistoryForest reconstruction now borrows shared Arc-backed deltas along the
chosen first-parent lineage instead of deep-cloning all historical property
values. GraphSnapshot reconstruction owns one working state; updates clone touched
new values, not the entire growing snapshot at every version. Node membership uses
one generation/order index, compacted at the end, preserving insertion order,
duplicate-input semantics, removal/re-addition and sparse original IDs without
rebuilding a full membership set at each scalar addition.

Public immutable apply_delta still returns an independent copy. Its payload
implementation is shared with the owning reconstruction path. The public
reconstruct_from_deltas signature remains unchanged. Source snapshots/deltas,
lineage choice, graph direction, logical deltas/hashes, temporal clocks and write
publication remain unchanged; this is a recorded-state read optimization.

Native contract tests compare every prefix against sequential immutable application
with duplicate IDs, removals/re-additions, unsets/null, edge changes and IEEE
values. A 10,000-scalar-add version case verifies stable membership/order without
an expensive reference fold. Existing history/owned snapshot/archive/branch tests
cover the actual HistoryForest caller.

## Release evidence and trade-offs

`benches/recorded_reconstruction.rs`; receipt:
`notes/benchmarks/recorded-reconstruction-2026-10-26.json`.
Three alternating pairs, same prepared inputs, single runner, Rust requested
allocation instrumentation. Compared to the previous sequential immutable fold:

- 10k initial nodes / 1k property versions: median **1.025 s → 1.722 ms**;
  allocation calls **29,533,502 → 31,004**, requested bytes **4,180,088,896 →
  4,623,432**. These are cumulative requested allocations, not RSS.
- 5k scalar node versions: median **233 ms → 0.308 ms**, allocation calls
  **15,010 → 24**. The membership/order index increases measured peak additional
  requested live bytes **233,688 → 274,448** in this fixture; the bounded extra
  index removes repeated membership work. Ordinary mutation latency is not measured.

No whole-engine, FFI, universal latency/RSS or graph-size SLA is inferred.
Final native/Python/docs/Clippy totals are in priority-slices-2026-10-26.md.
