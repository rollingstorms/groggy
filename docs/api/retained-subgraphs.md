# Retained subgraph membership

Ownership update: [default selections are immutable](default-immutable-views.md).
The earlier live-scope details below apply to explicit native execution scopes
and `graph.connected_view()`; default bound writes require a matching source.

Stored subgraph references identify retained node/edge membership. Deactivating a working node or edge does not edit a stored selection. State switching and hard reset replace selected membership and property mappings while retaining the graph's pool, allocated IDs, endpoints, property columns and stored selections.

`create_meta_node(reference)` checks that the referenced selection exists before allocating a node or recording a change. Storing the same membership/type reuses its reference; a reference-hash collision rejects instead of replacing different retained content. The old low-level `update_stored_subgraphs_remove_node/edge` methods are deprecated compatibility no-ops: deactivate through Graph instead.

Collapse preserves original base/meta identity. Extract keeps original nodes active; collapse removes them from the selected graph after aggregation and meta-edge creation. Strict and allow-missing paths apply the same node strategy. Both retain the stored original membership for expansion and later branch restoration.

## Current boundary

Qualified group revisions retain immutable values and topology. Expanded meta-node
subgraphs bind to that retained source, including identities removed from the
working graph. Legacy membership-only groups require an explicit source chosen
with `pin_legacy_group`; their missing historical values cannot be invented.

A temporal snapshot can expose its stored reference as a typed property. Materializing it as a separate graph currently rejects references without their backing pool instead of returning an unusable object. The durable journal and active JSON export still reject pooled/meta contents; persistence needs an explicit transferable representation.

Restoring a state currently appends its restored property values into the retained columns. Repeated switching can therefore grow those columns without adding data versions. Reuse of retained property locations or bounded reconstruction storage remains part of item 12; this repair makes no allocation/performance claim.

Verification: `tests/retained_subgraphs.rs` checks retained membership, column indices, endpoints, branch-only allocations, resets, absent referents and detached materialization. `tests/test_retained_subgraphs.py` exercises strict/lenient extract/collapse, branch switching and nested meta identity through the built extension. Collapse is still a legacy manually grouped operation, not a newly supported durable mixed-batch operation.
