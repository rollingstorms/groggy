# Composite table routing and metadata sharing — 2026-10-25

Existing 27.1 / AR7.10; no new public APIs or checklist rows.

## Closed operation inventory

Native sort_by/sort_values, group_by, pivot_table and melt share one component
resolver. All operands are considered together. Sorting/reshape preserve the
historical node preference for shared-only columns; grouping still rejects an
ambiguous shared key and directs callers to the explicit component. Unknown or
cross-component sets reject. This repairs edge melt with edge identifier columns
and a value column also present on nodes, which previously selected nodes and
failed. Policy, structural reshape checks, row/source provenance and untouched
component behavior are retained.

## Ownership detail and correction

BaseArray already shared its immutable value vectors before this change.
The new GraphTable Arc ownership shares the entire node/edge table object,
including schema/column handles, above that existing buffer sharing. It avoids
per-clone/per-group table metadata copies, not an earlier O(groups * edge rows)
value-copy cost. The initial progress explanation overstated that cost; this
record and the API/benchmark descriptions correct it.

Native nodes_mut/edges_mut detach shared table metadata through its existing
Clone behavior; column value buffers remain shared until existing COW or
replacement operations change them. Shared component references retain original
resources until released. Detached BaseTable metadata follows existing Clone
rules, including not copying streaming handles. into_components moves unique
objects or clones shared metadata; explicit edge-ID assignment stays isolated.
No public signatures changed. This does not add COW to Graph.read_view captures
or remove Python component-wrapper construction costs.

## Evidence

Six routing cases cover node/edge/shared/mixed operands, edge melt repair,
policy, structural validation and source identity. Five sharing cases cover
component pointer identity, existing column-buffer identity after metadata
detach, independent mutations, shared untouched groups/head/sort results,
owned component extraction, nullable edge assignment and checked publication.
API examples: `docs/api/graph-table-components.md`.

`benches/graph_table_component_sharing.rs` records five current-cost observations
at 1k/10k/100k nodes plus chain edges. Clone/drop uses 10000 iterations per
observation and reports amortized seconds to exceed clock resolution; its drop
cost is included. Grouping (100 groups) and first mutable metadata access time
one output and exclude destruction. Fixtures, prepared clones and assertions
are excluded. Pointer checks certify sharing, not RSS. Receipt:
`notes/benchmarks/graph-table-component-sharing-2026-10-25.json`.
No matched speedup, whole-graph capture COW or FFI budget is claimed.

## Fixed remaining inventory for 27.1 / AR7.10

1. Node-subset edge retention: decide induced edges versus independent table
   bundle semantics. Existing head/tail/slice and node-group edge retention
   remains unchanged; no unanswered owner choice is treated as approval.
2. Composite metadata / rectangular Table-trait interpretation. The existing
   aggregate counts, primary node schema and node shape remain a compatibility
   surface; callers can use nodes/edges for rectangular operations.
3. Existing unsupported legacy converter entry points: Python GraphTable
   to_nodes/to_edges/to_subgraphs remain NotImplemented. Their context/identity
   contracts need to be completed; the existing explicit to_graph materializer
   and component access remain available. These stubs predate this pass and
   remain within the original conversion scope, not newly added requirements.

Legacy connected Subgraph/accessor object behavior remains in 26.1. Full retained
pool/history portability remains in 07.3/10.3/12.2. Table output pin migration,
source writeback, generic joins/rows/collections, specialized merges, typed
bundles, schema validation and the routing/sharing families above are complete
within their recorded boundaries.


### Composite routing and shared component metadata — 2026-10-25

Closed native sort/group/pivot/melt operand routing and GraphTable component
metadata sharing under 27.1/AR7.10. Edge melt now respects its identifier component
when values have shared names. Existing sort/reshape node preference and grouping
ambiguity rejection remain. Clones/groups share table/schema objects above the
already shared BaseArray value buffers; mutable access detaches metadata and
retains the existing column COW/replacement rules. Initial progress commentary
overstated prior value-copy costs; evidence/docs explicitly correct that claim.

Evidence: `graph-table-component-conformance-2026-10-25.md`; 6 routing / 5 sharing
native cases, including component and buffer pointer checks, independent edits
and source publication. Gate: **1,131 native passed / 51 ignored; 1,861 root Python
passed / 1 skipped; 127 executable docs passed; strict all-target Clippy passed**.
Receipt: `notes/benchmarks/graph-table-component-sharing-2026-10-25.json`.
Clone/drop timing is amortized over 10000 iterations to exceed clock resolution;
no matched speedup, RSS, whole-graph COW or FFI latency guarantee.

27.1 now has a fixed explicit remainder: node-subset edge policy; composite
metadata/rectangular Table interpretation; existing unsupported to_nodes/to_edges/
to_subgraphs converter contracts. These pre-existing facade contracts remain
within the original parent, not new checklist rows. Legacy object migration stays
in 26.1; full retained-pool/history portability remains separately tracked.
Inventory: **192 checked / 76 unchecked**, unchanged by adding no rows.
