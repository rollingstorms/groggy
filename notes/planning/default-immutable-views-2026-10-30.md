# Default immutable view ownership — 2026-10-30

Completion receipt for existing **26.1 / AR4 / AR7.5**. This finishes the default
binding migration after the earlier captured-kernel, selected export and checked
write slices. No new checklist rows are introduced.

## Implemented boundary

- Default Python Subgraph results retain properties, endpoints, direction,
  selected membership and retained group sources at creation. Reads, derivations,
  tables, matrices, visualization and owned copies use that same capture. Lazy
  component arrays retain it even when empty or before the first item is accessed.
- Native Subgraph and specialized wrappers support `.freeze()` without losing
  filter/component/center metadata. Default Python return factories freeze once;
  derived trait objects preserve their source rather than rebuilding from current
  owner membership. Explicit native constructors remain execution scopes.
- Frozen specialized filter matching shares the existing criteria kernel;
  combinations/complements/statistics/additional criteria keep their source.
  Filter reapplication explicitly returns a fresh capture. Frozen neighborhood
  expansion, exact hop rows and merges keep the captured source, with checked
  hop overflow. Earlier views remain unchanged.
- Explicit bound bulk writes, composition, algorithm execution and pipeline
  execution validate history, branch, accepted head and exact source before using
  the owner. Existing pipeline staging/atomicity and owner-wide internal algorithm
  attachment contracts remain intact. Algorithms return a refreshed frozen result;
  bulk setters leave the original read unchanged. Collection composition resolves
  all source scopes before its existing sequential effects.
- Meta expansion uses the retained group revision directly, without recapturing
  the whole current graph. Legacy membership-only references require an explicit
  source instead of inventing old values. Current and captured group catalogs
  retain their distinct lifetimes.
- `Graph.view()` shares one canonical default capture while a caller holds it;
  explicit `read_view()` captures retain distinct provenance. Arc membership and
  weak payload/capture/Python caches avoid persistent snapshot pins. Complete
  read revisions include property/endpoint/catalog changes, restoration, branch
  and head. Owner operations use a separate lightweight live execution cache.
- Default node/edge adapters are immutable and provide source-bound tables.
  Their `ids` is a property. Neighbor arrays now use checked integer IDs rather
  than a float round trip. Captured filtering, neighbors and matrix projection
  release the GIL over immutable input/owned transfer types.

## Compatibility and costs

`graph.connected_view()` explicitly exposes the previous live-owner contract.
Default-facing results returned from live execution scopes are captured on return.
Native `Subgraph::new` remains an explicit live execution constructor. Owning
Graph accessors still offer working mutations; readonly adapters do not hide a
mutable fallback. The owner-bearing Python Subgraph remains thread-bound;
`pin_view()` exposes its thread-safe, owner-free GraphView.

The API change is intentional and documented: earlier default reads no longer
follow owner writes, repeated effects require a refreshed result/source, and
selection entity adapters use readonly captured methods (`ids`, not `ids()`).
Existing compatibility-scope tests remain alongside new immutable defaults.

Cold capture still materializes full source data; a small selected view can retain
that full source. Warm cache access avoids graph-size work. Physical snapshot and
owned export costs remain linear. Benchmark evidence and final gates are recorded
below after the verified release run.


## Final verification and measured costs

- Full `cargo test --offline --locked`: **1,514 passed / 54 ignored**.
- Root `cargo clippy --offline --locked --all-targets -- -D warnings`: passed.
- Release extension rebuilt and atomically installed in-tree; three existing FFI warnings remain.
- `python notes/scripts/verify_immutable_view_migration.py`: **2,574 passed / 1 skipped**, plus **255 executed documentation examples**.
- Scoped Rust formatting, Python compilation and `git diff --check`: passed.

Quiet release measurements (1k / 100k undirected chains, one f64 property per node):
cold full capture **0.776 / 147.879 ms**; warm held default view **68 / 66 ns**;
explicit connected view **38 / 38 ns**; warm owner degree **0.610 / 0.614 µs**;
retained single-node meta expansion **0.542 / 0.502 µs**. Warm samples hold their
read objects; temporary unowned captures can expire. Construction is excluded;
wall and process CPU are recorded separately. This is not a whole-engine latency,
RSS or production scalability guarantee. Raw evidence:
`../benchmarks/default-immutable-views-2026-10-30.json`; script:
`../../benches/default_immutable_views.py`.

Fixed checklist: **211 checked / 57 open → 214 checked / 54 open**, still **268
rows**. The three closed rows describe this implementation and its overlapping
architecture exits. Existing compound topology/effect publication (09.4b.2/AR3)
and legacy query migration (21.2) remain priorities; no new rows were added.
