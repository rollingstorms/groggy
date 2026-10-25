# GraphTable presentation schema and cache ownership — 2026-10-25

Closes the remaining native display/streaming **table-schema** boundary in
27.1/AR7.10, supporting AR7.6. No new checklist rows or broad display/algorithm
completion claim. The unresolved to_subgraphs contract remains in 27.1.

## Projection contract

The core graph bundle owns two rectangles. Its native DataSource adapter is now
an explicit presentation projection, owned by the private graph_streaming module:

- entity_kind identifies a node or edge row.
- Node columns are prefixed node_; edge columns are prefixed edge_.
- Node rows precede edge rows; the opposite component's cells are null padding.
- Headers, schema, column types and width describe that exact projection, using
  the authoritative BaseTable component schemas.
- Windows can cross the node/edge boundary. Empty/out-of-range/overflow-sized
  requests preserve the projection schema and requested offset.

Previously total_cols counted a name union, schema prefixed both components,
and windows returned component-local headers/rows and stopped at the boundary.
Those three incompatible representations now agree. Shared property names no
longer imply one shared physical column. No graph identity or history is
published by presentation, and the projection is not a portable identity-aware
interchange or the core GraphTable.shape contract.

## Cache identity

Each native graph-table presentation has a process-local identity. Clones share
the immutable components and ID. Every derived graph-table constructor gets a
fresh ID; nodes_mut/edges_mut refresh it before handing out mutable access.
Policy changes and explicit edge-ID assignment also refresh it. A fixed cache
version is paired with that identity; it is not a graph state/version number.
This prevents same-shape/same-version derived branches from sharing stale
windows, without content hashing or scans in cache-key construction. IDs are
private, constant-size and never serialized as graph/entity/history identity.

## Verification and performance boundary

Four native tests cover tagged widths/headers/component values/types, windows
across the boundary, empty/overflow requests, stable clone keys, distinct fresh
captures/derived branches, invalidation on both mutable components/policy/ID
assignment, and retained original values. Full gate: **1,161 native passed /
51 ignored; 1,890 root Python passed / 1 skipped; 134 executable docs passed;
strict all-target Clippy passed**. In-tree extension rebuilt.

Current release receipt: `notes/benchmarks/graph-table-streaming-identity-2026-10-25.json`.
Same induced 100-group fixture as the preceding slice; at 100k nodes median
grouping is about 28.2ms, amortized clone/drop about 13.3ns, and first mutable
metadata access about 2.3 microseconds. Five observations, existing timing/drop
boundaries and assertions are recorded in the receipt. These are current costs,
not a speedup claim. No window-throughput, RSS, Python/FFI, production-scale or
whole-graph capture claim. Cache IDs add one private u64 per graph-table wrapper
and refresh through an atomic counter; no new per-clone allocation is introduced.

Legacy native graph-data/layout methods of DataSource were moved with the
adapter but are unchanged and are not certified by this projection repair.
The checked to_viz_data_source/.viz paths remain the graph-validation boundary.
Other display configurations, native legacy graph adapters and viewer controls
remain in AR7.6/28.15. Full connected views and pooled/history portability retain
their original scope in 26.1 and 07.3/10.3/12.2.
