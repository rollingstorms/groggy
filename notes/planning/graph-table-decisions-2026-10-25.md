# GraphTable owner decisions and implementation — 2026-10-25

The owner explicitly confirmed all three recommendations in this chat:

1. Node subsets retain only edges whose endpoints are both selected.
2. Report separate node/edge shapes; rectangular operations use the components.
3. to_nodes/to_edges return table components. New graph ownership remains an
   explicit to_graph operation.

These answers supersede the earlier pending selection/facade/converter choices.
The separate to_subgraphs meaning is still being clarified; these three answers
do not silently answer it. Existing 27.1/AR7.10 scope, no new checklist rows.

## Implementation and compatibility

Native head/tail/slice and node grouping share one checked induced-edge adapter.
Node grouping builds one node-to-group map and dispatches edges once; crossing
edges are excluded. Parallel edges, loops, physical IDs/order, typed properties,
policy and exact source row identity are preserved. Unchanged whole edge
components remain shared; projected rows use BaseTable.select_rows. Malformed or
duplicate structural IDs and dangling input endpoints reject. Filter retains
its existing both-component predicate rule, then enforces internal-edge closure.

GraphTable no longer implements the rectangular Rust Table trait. Its graph
operations are inherent methods. Rectangular generic code must use nodes()/
edges(); head/tail/slice now return GraphResult because invalid input can fail.
Native shape returns GraphTableShape with nodes/edges fields, and column_names
returns separate component schemas. Python shape() returns a dictionary and
columns exposes the two name lists. nrows/len retain aggregate entity counts;
ncols retains distinct-name count, neither defines a rectangle. These public
contract changes are deliberate consequences of the owner's accepted design.

Python and native to_nodes/to_edges return independent component wrappers over
existing shared value buffers and retained source captures. No graph/IDs/history
is created; owned edits do not modify the original. Checked mixed source
publication remains available. Python now exposes checked slice and group_by
facades, with graph/row logic in Rust and GIL release around native work.

The new range tests also exposed a pre-existing BaseTable slice bug: it passed an
end index to BaseArray.slice, which expects a length. Middle/reversed ranges
could violate declared column lengths. It now passes end-start and retains
aligned source IDs for those ranges.

## Evidence

Seven new native cases cover sparse IDs, loops/parallel edges, bounds/empty
schemas, group partition/closure, separate shapes/schemas, exact mixed source
publication, malformed input and middle/reversed base range identity. Existing
sharing/routing tests were updated to the newly accepted induced policy, while
retaining their component-copy and ambiguity checks. Nine Python cases exercise
the public facade and ownership/writeback. Four executable API examples are in
`docs/api/graph-table-components.md`.

Full gate: **1,157 native passed / 51 ignored; 1,890 root Python passed / 1 skipped;
134 executable docs passed; strict all-target Clippy passed**. In-tree extension
rebuilt; three pre-existing unrelated FFI warnings remain.

`benches/graph_table_component_sharing.rs` now asserts induced grouping. Current
receipt: `notes/benchmarks/graph-table-induced-groups-2026-10-25.json`; five release
observations, 100 groups on 1k/10k/100k chains. At 100k nodes median grouping is
about 31ms. Group labels are row modulo 100, so every chain edge crosses groups
and each output edge table is empty. Validation, mapping and row projection are
timed; fixtures/assertions/output destruction excluded. Clone/drop remains
amortized over 10000 iterations. The earlier bundle-retention receipt has
different semantics and is not a matched speedup comparison. No RSS, whole-graph
capture COW, Python/FFI or production-scale guarantee is claimed.

## Remaining existing scope

The to_subgraphs contract remains open. Native display/streaming table schemas are now consistent through an explicit
tagged projection; see `graph-table-streaming-2026-10-25.md`. Legacy native
graph-data/layout compatibility methods remain separately scoped in AR7.6/28.15. Broader connected-view migration and portable
pool/history storage remain 26.1 and 07.3/10.3/12.2. Effective-time reconstruction
remains 16.2; no implementation claim is made for it by this table slice.
