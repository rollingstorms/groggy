# Shared pinned read-state foundation — 2026-10-22

AR7.5a introduces one immutable state payload shared by current read captures and
the existing TemporalSnapshot facade. **Ordinary Subgraph reads are not yet
migrated**; AR7.5/AR4/26.1 remain open until all reads, nested views, algorithms and
write targets are wired consistently.

## Implemented boundary

- Graph::pin_current_state captures active nodes, edges, endpoints and exact
  current typed attribute values, including uncheckpointed edits. It creates no
  history record, checkpoint or mutation delta.
- Capture projects GraphSpace indices in attribute-first order, borrowing the
  pool once. Dangling active value indices/endpoints fail explicitly rather than
  turning into missing cells. Inactive allocations are not part of live membership.
- PinnedReadState owns an Arc payload: snapshot values, active membership and
  indexed adjacency. Clones share this immutable data; no writable graph reference
  is exposed. Captures survive source mutation/deactivation/destruction and can
  cross read-worker boundaries. This does not implement concurrent graph writers.
- TemporalSnapshot keeps lineage metadata and delegates reads to that payload.
  Its duplicate cloned attribute maps are removed. Historical imports and method
  signatures remain stable; snapshots reject dangling edge endpoints.
- Snapshot neighbor occurrences retain parallel facts, sort deterministically,
  preserve direction/isolates and count an undirected self-loop once, matching the
  previous temporal snapshot convention. Global live-neighbor consolidation is
  still separate work; this does not certify every caller's semantics.

## Identity and cost

A working capture's underlying GraphSnapshot.state_id identifies checkpoint
ancestry, not a newly recorded version of dirty values. Do not identify/collapse
read states by that ID or by GraphSpace topology version: attribute-only writes
currently leave the topology version unchanged. Separate captures after those
writes must retain different property values.

Capture owns current values and is proportional to scanned index bindings plus
active topology and adjacency sorting; it is not a constant-time graph clone.
Cloning an existing read pin shares the payload. No global cache is introduced.
Stored meta/subgraph references remain typed references in captured attributes;
this slice does not claim portable retained group expansion, writeback, or arbitrary
timestamp reconstruction. Those retain their existing checklist/workbook gates.

## Next migration

Share the pin across related views while keeping a separate working-graph mutation
target. Route graph/table reads, nested selections, algorithms and cache identity
through the pin. Keep writes on the owner and preserve the old pin's observations.
Complete this migration before describing ordinary graph views as immutable.

## Verification

Native tests cover dirty/current captures with unchanged topology version, null
versus absent, i64 exactness, post-removal and post-owner-drop reads, cheap shared
clones, directional/parallel/self-loop/isolated topology, historical facade reuse,
dangling values and read-worker transfer. Existing temporal/property/persistence/
mutation/Python suites are rerun on a rebuilt extension.

Executed: full locked offline native suites, strict all-target core Clippy (clean),
rebuilt FFI and temporal/property/persistence/mutation Python gate: **92 passed**.
Idle release Criterion receipts cover 1k/10k-node capture and clone costs;
`notes/benchmarks/read-state-2026-10-22.md`. Native capture is 0.574/7.63 ms on
these fixtures versus approximately 10.2 ns to share an existing pin. This is not
an FFI latency or constant-time capture claim.
