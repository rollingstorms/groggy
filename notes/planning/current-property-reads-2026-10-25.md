# Current property reader consolidation — 2026-10-25

Existing scope: 07.4 / 33.2 / AR5. No new checkbox row.

## Fixed implementation inventory: twelve existing entry points

| Node reader | Edge reader | Status |
| --- | --- | --- |
| get_node_attr | get_edge_attr | Complete |
| get_node_attrs | get_edge_attrs | Complete |
| get_node_attrs_bulk | get_edge_attrs_bulk | Complete |
| get_nodes_attrs | get_edges_attrs | Complete |
| _get_node_attribute_column | _get_edge_attribute_column | Complete |
| _get_node_attributes_for_nodes | _get_edge_attributes_for_edges | Complete |

All twelve now share `src/api/graph/read_attributes.rs`, preserving public names,
argument/result types and native/Python import paths. The Graph coordinator loses
384 lines; the focused module owns 256 lines with common validation and value
resolution. This is native business logic; FFI remains unchanged.

Scalar/row methods reject inactive IDs. Aligned columns retain None for inactive
IDs, requested order and repeated IDs. Bulk row maps retain map semantics. Present
Null and absent properties remain distinct in native data; Python getters may
project both to None, while typed snapshots preserve the distinction.

A current binding referencing a missing column or out-of-range retained index now
returns InvalidInput with entity kind/ID, property and index. It cannot silently
turn into absence. Requested projections validate their requested rows/keys only;
all-property row reads validate every row binding. Reads do not record changes,
consume IDs, create checkpoints or invalidate immutable pins.

Column paths resolve the storage column once. Bulk row projections borrow the pool
once and inspect only requested indices rather than cloning all row indices.
All-property reads still enumerate the row's indices. This does not certify every
legacy consumer that reads raw pool data or discards Result errors; broad integrity,
view and index audits stay open.

## Verification

Three native regressions exercise all twelve entry points under missing-column
and out-of-range corruption, null/absence, inactive IDs, exact fractional f64,
order/repeats, empty requests, selected-scope isolation and unchanged retained pins.
Full gates: 1,020 native passed / 52 ignored; 1,730 root Python passed / 1 skipped;
88 executable documentation examples; in-tree FFI build; strict all-target Clippy.

## Bounded optimized observations

`benches/current_property_reads.rs`, five observations, 32 user property columns
per kind. One f64 scalar per entity, a two-key row projection (one absent key),
or one aligned column. Setup/final value verification excluded. Projection includes
map-shape checks and ordered output extraction, so modes are distinct workloads.
Median milliseconds:

| Entities | Kind | Scalar collection | Row projection | Column |
| ---: | --- | ---: | ---: | ---: |
| 1,000 | edge | 0.1112 | 0.3084 | 0.0946 |
| 1,000 | node | 0.1093 | 0.3073 | 0.0937 |
| 10,000 | edge | 0.9191 | 2.6095 | 0.8115 |
| 10,000 | node | 0.9165 | 2.5652 | 0.8033 |

Receipt: `../benchmarks/current-property-reads-2026-10-25.json`. These are native
current-cost observations, not a matched speedup, allocation/RSS, broad graph
scale or per-call FFI guarantee. Existing larger performance gates remain open.
