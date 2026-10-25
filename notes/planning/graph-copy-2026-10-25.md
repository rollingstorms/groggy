# Prepared active graph copying — 2026-10-25

Existing scope: 09.4b.2, 05.3, 07.4 and AR7.10. No new checkbox row.

## Implementation

Removed the manual add_graph loop that ignored attribute errors and could consume
IDs before rejecting source content. Both modes now capture validated active values
and endpoints, stage the complete copy and use the prepared storage executor.
Publishing accepts one version; manual leaves pending changes, including when the
destination already has dirty manual edits. Empty sources do nothing; dirty
publishing destinations reject nonempty copies. Source state/history is unchanged.

Fresh IDs follow sorted source IDs and destination allocation counters, not active
counts. Typed NodeArray and EdgeArray values are remapped across both property
kinds, including forward references and repeats. The canonical storage helper is
shared with table materialization and consumes owned buffers. Plain integer lists
and JSON stay opaque. Missing/inactive typed references reject. Managed identities
require upserts, pooled/meta contents require portable pool support, corrupt value
indices/endpoints reject and non-null contains_subgraph metadata cannot bypass this
rule. Destination direction/configuration stays authoritative as before.

Source capture stages owned values once and scans current attribute-index bindings;
reference rewriting reuses owned vectors. It does not build an adjacency cache.
Deterministic node/edge ID assignment adds sorting cost. General read methods,
portable pool copying, history copying and pinned connected provenance remain open.

## Verification

Five native regressions exercise both modes, reference kinds on both entity kinds,
allocation gaps, parallel edges/loops, opaque/typed values, temporal/null/f64 values,
corrupt source columns/kinds, absent references, managed IDs/meta nodes and dirty
boundaries. Ten Python cases cover active values/history/topology and rejection
without consumed IDs. Existing table reference tests verify the shared helper.

Full gates: 1,015 native passed / 52 ignored; 1,730 Python passed / 1 skipped;
87 executable documentation examples; in-tree FFI build; strict all-target Clippy.
The initial benchmark compile called a nonexistent edge-column convenience method;
it was corrected to the canonical bulk setter before all-target and release gates.

## Bounded release cost

`benches/graph_copy.rs`, five observations per case. Source sparse IDs, f64 scalar
and 16-element vector properties plus typed node/edge references. Complete capture,
validation, remapping, installation and optional publication measured; fixtures and
exhaustive output checks excluded. Median milliseconds:

| Copied nodes | Manual | Publishing |
| ---: | ---: | ---: |
| 1,000 | 3.3726 | 7.6005 |
| 10,000 | 38.8330 | 88.8583 |

Receipt: `../benchmarks/graph-copy-2026-10-25.json`. This is current-cost evidence,
not a matched speedup. History retention differs between manual and publishing
modes. No RSS/allocation, dense topology, portable-pool or FFI latency claim.
