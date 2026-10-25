# Checked BaseTable entity adapters — 2026-10-25

Existing 27.1/AR7.10 table conversion scope; no checklist rows added. This closes
the BaseTable converter family, separately from the three unresolved GraphTable
graph-accessor/subgraph converter stubs.

## Implementation

Native BaseTable.to_nodes(uid_key) previously accepted only the literal node_id
name despite its explicit key argument and rename TODO. It now renames a chosen
integer key using the existing column rename implementation, retaining order and
shared value buffers. Missing keys, canonical-name collisions and malformed or
duplicate IDs reject. BaseTable.to_edges now uses existing EdgesTable alias
normalization plus native edge validation: unique IDs, checked endpoints,
parallel edges and loops allowed. No outside graph/node membership is inferred.

Thin Python BaseTable.to_nodes(uid_key="node_id") / to_edges() methods expose
these native adapters, release the GIL and use standard graph error translation.
They return NodesTable/EdgesTable, allocate no graph IDs and publish no history.
Existing from_base_table wrappers remain permissive schema adapters for import
inspection and repair; checked conversion is an explicit boundary.

Provenance handles this centrally: retain writable row binding only when kind,
IDs and edge endpoints still describe the captured source. Changed identity,
kind or topology keeps source captures as read-only context. Derived/unbound
rows never acquire inferred writeback ownership. Compatible renames preserve
source state and exact row identity, enabling one checked explicit update.

## Verification

Seven native cases cover custom names/order/buffer sharing, strict rejection,
optimized IDs, existing edge aliases, loops/parallel edges, null/duplicate
identity, exact source publication once, changed identity/kind/endpoints,
retained read-only context and empty schemas. Eleven Python cases cover both
adapters and the corresponding source-write behavior. Three API examples are in
`docs/api/checked-table-conversions.md`; Python signatures are in _groggy.pyi.

Gate: **1,150 native passed / 51 ignored; 1,881 root Python passed / 1 skipped;
133 executable docs passed; strict all-target Clippy passed**. The final focused
native suite also verifies the one-state publication assertion.

Explicit conversion adds linear structural identity validation and, for a bound
edge table, captured-endpoint checks. Column payloads remain shared; native
pointer tests certify that boundary, not RSS or a speedup. No benchmark/FFI
latency improvement is claimed for these validation adapters.

## Existing open boundaries

GraphTable.to_nodes/to_edges currently promise graph accessors and to_subgraphs
promises SubgraphArray; those remain unsupported pending ownership/identity
contracts. GraphTable node-subset edge retention and rectangular metadata also
remain open. No implicit direction, topology, graph ownership or pooled/history
materialization is added by a BaseTable structural conversion.
