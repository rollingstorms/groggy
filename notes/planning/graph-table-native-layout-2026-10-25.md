# Native graph-table layout membership and direction — 2026-10-25

Existing 28.15 / AR7.6 native visualization compatibility scope; no checklist
rows added and no blanket algorithm certification.

GraphTable's native DataSource.compute_layout previously delegated to EdgesTable,
whose node set is inferred from endpoints. It omitted every isolated node,
including the entire one-node/zero-edge graph. The adapter now uses the complete
checked table-to-GraphDataSource capture, preserving node row order and original
IDs, then delegates layout. The infallible DataSource trait returns an empty
result for rejected input; callers needing error details must use the explicit
fallible to_viz_data_source boundary. This is not a new error-capable layout API.

The native compatibility metadata also stops guessing direction from reciprocal
edges. Table bundles have no direction setting, so that path uses the established
undirected default; to_viz_data_source(true) remains the explicit directed path.
Removing the reciprocity guess also removes its nested edge scan. This does not
certify every other metadata field, legacy graph reader, layout control, solver
quality or fallback behavior.

Two native regressions verify one node, a connected pair plus an isolate,
original layout IDs/order, and explicit versus default direction for asymmetric
and reciprocal edges. Full gate: **1,163 native passed / 51 ignored; 1,890 root
Python passed / 1 skipped; 134 executable docs passed; strict all-target Clippy
passed**. In-tree extension rebuilt; existing FFI warnings unchanged.

Current release receipt: `notes/benchmarks/graph-table-native-layout-2026-10-25.json`.
The grid case uses 1k/10k/100k nodes, chain edges plus one isolated node and no
property columns. Five observations time complete checked capture plus grid
position construction, excluding fixture construction/assertions/output
destruction. All node positions, including the isolate, are asserted. Median
100k cost is about 43ms. This additional complete-node validation/capture is
linear work and allocates owned visualization entities; it is not a speedup
against the old incorrect endpoint-only layout, an RSS/FFI claim or production
latency promise. Other existing cases retain their own described fixtures.

The current 27.1 remaining inventory stays to_subgraphs grouping/ownership,
awaiting the owner answer. Effective-time reconstruction, wider legacy view
migration and other visualization controls retain their separate existing scope.
