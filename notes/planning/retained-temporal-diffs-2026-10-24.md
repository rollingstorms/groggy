# Retained temporal diff completion — 2026-10-24

Completes the existing `diff_nodes` / `diff_edges` family under **30.1 / AR7.8**,
including the shared TemporalDelta property comparison. No checklist rows or new
capabilities were added to the plan. Existing helper implementations were
incomplete, and the existing property/state comparison requirement covers these
repairs. The broad execution/history parents retain their original exit criteria.

## Accounting correction

The owner observed that work appeared to expand without reducing the open list.
There were two causes: audits found real incomplete implementations, and each
repair was recorded as another checked slice while broad overlapping parents
remained open. The flat checklist header was also stale at 158 checked despite
189 checked entries. It now matches the actual **189 checked / 79 unchecked**
inventory. The 79 comprise 65 numbered entries (nine explicitly deferred) plus
14 overlapping architecture entries; numbered publication parents also overlap.
This is not an independent-task count or completion percentage.

This pass updates existing requirements/evidence. New capabilities must be
identified as scope additions. A fixed inventory is used for the current family:

- Completed previously: snapshot_at, decay.
- Completed now: diff_nodes, diff_edges.
- Remaining: window_aggregate, filter, mark_changed_nodes, window.

Thus remaining temporal primitives fell **six → four**, out of the same eight
primitives that already existed. The broader 30.1 parent also contains unrelated
execution/intermediate/pin requirements, and 16.2 still requires effective-time
reconstruction; this pass does not falsely mark those complete.

## Shared logical diff kernel

`TemporalDelta::compute` previously cloned membership sets and left property
records empty. It now borrows immutable sets/maps, compares actual properties on
surviving nodes/edges, owns changed old/new values and sorts membership IDs and
property records by entity ID/name. Summary/is_empty/affected entity APIs consume
those records. Absence and explicit null differ. Typed equality uses existing
AttrValue semantics, including vectors/structured values and float representation.
New/removed entity properties are covered by membership changes, not duplicated
as modified-property records. Results are net state differences, not intermediate
events. Reverse and sibling-branch comparisons in one history are supported.

A private shared in-process token links HistoryForest and its TemporalSnapshots.
Equal numeric state IDs from different graphs are not accepted as provenance.
Snapshots preserve the token on clone and remain independent of a dropped owner.
No storage/wire schema changes or history clones were introduced. A separately
loaded/copied graph has a distinct instance. Node/edge steps additionally require
both inputs to belong to the invocation owner's history. Their existing context
fallback still resolves reference/current retained snapshots.

The delta representation does not contain before/after edge endpoints. A changed
endpoint on a surviving edge now rejects explicitly rather than yielding a false
empty/partial delta. Existing 04.2/13.3 already tracks endpoint-change support.

## Checked masks and effects

Both steps support atomic property execution. Existing added/removed names remain;
additional `<prefix>_nodes_modified` / `_edges_modified` maps mark surviving IDs
with property changes. Masks use value 1. Comparisons cover full retained owner
states and do not clip historical IDs to the current selection. Explicit
attachments use the existing active owner-ID boundary, including active IDs
outside a selected view; inactive removed IDs cannot be written as live entities.

Private masks are prepared before any destination is installed. Missing, wrong-
typed and foreign snapshots leave prior variables intact. Context.delta now uses
cooperative cancellation throughout entity/property comparison. Mask loops and
pre-installation check cancellation too. Sorting, cloning/comparing large values
and allocation have no bounded interrupt-latency promise. A failed enclosing
atomic run discards earlier staged properties. Snapshot reads ignore those staged
writes, and no diff read creates a mutation or checkpoint.

Unsupported-effect fixtures now use the still-uncertified temporal.filter with
valid parameters, keeping empty-loop/for-each preparation checks meaningful after
diff certification.

## Verification

- `CARGO_INCREMENTAL=0 cargo test --offline --locked --quiet`: **993 passed,
  52 ignored**, including eight new native regressions.
- Native tests cover actual logical values, absence/null/type changes, stable
  names/IDs, unchanged NaN semantics, membership and reverse deltas, cross-history
  numeric ID collisions, sibling branches, mid-traversal cancellation, unsupported
  endpoint changes, foreign-owner output preservation and historical removed IDs.
- New root Python integration file: **11 passed**. Persistent/transient and
  forward/reverse maps, staged-write exclusion, one publication, equal-state no-op,
  missing variables/rollback, inactive removed writeback and empty-selection
  full-owner behavior. Combined focused builder/for-each suite: **138 passed**.
- Full root Python gate: **1,645 passed, 1 skipped**; **78 executable documentation
  examples passed**, including three new diff examples. Owner's separate benchmark
  test remains excluded from this gate and unchanged.
- Strict all-target core Clippy passed. In-tree FFI rebuild passed with the same
  three pre-existing unrelated warnings. FFI remains translation only.

## Bounded release receipt

`benches/temporal_diff_steps.rs` and
`../benchmarks/temporal-diff-bundle-2026-10-24.json` record six workloads with five
observations each. Native optimized kernels compare retained ring graphs with all
node/edge scalar properties changed. Fixture/snapshot/variable preparation and
validation are excluded; provenance checks, comparison, sorting and private result
installation are included. Validate every old/new value or mask and unchanged
owner state. Median milliseconds:

| Operation | 1,000 nodes/edges | 10,000 nodes/edges |
| --- | ---: | ---: |
| Full logical delta | 0.8193 | 8.6977 |
| Node masks | 0.9049 | 9.4358 |
| Edge masks | 0.8286 | 9.4028 |

Each mask helper currently computes the full logical delta. Running both repeats
that work. There is no cross-step cache, matched speedup, allocation/RSS/FFI or
production-scale claim. Changed payload copies and stable sorting are material
costs. Commands:

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=python-groggy/target cargo bench --offline --locked --bench temporal_diff_steps --no-run
GROGGY_DIFF_PROFILE=notes/benchmarks/temporal-diff-bundle-2026-10-24.json python-groggy/target/release/deps/temporal_diff_steps-208068d49c3519ba
```

Public contract and examples: `../../docs/api/builder-temporal-diffs.md`.
The next existing component is window_aggregate: value-history semantics and
retained branch/index provenance must be correct before its box can close.
