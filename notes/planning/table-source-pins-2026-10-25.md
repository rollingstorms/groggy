# Frozen table source pins and checked writeback — 2026-10-25

Existing scope: 27.1 / 27.2 / AR7.10. No new checklist rows; broad parents remain
partial with their wider exits intact. Inventory remains 191 checked / 77 unchecked.

## Implemented inventory

| Existing family | Result |
| --- | --- |
| Frozen node/edge/bundle projection | Owned cell buffers plus shared GraphView source capture and immutable entity row IDs. Native and transferable FFI export paths agree. |
| Clone and column selection/drop/edit/rename | Share the capture and row binding; physical ID columns are not the identity authority. |
| Head/tail/slice, row selection/masks, filters, both sorts | Preserve exact row order/identity, including empty and repeated selections. |
| Grouped row selections, duplicate/drop-row filtering, missing-value filtering | Carry selected row identity. Repeated drop indices remove a row once. |
| Five generic joins: inner/left/right/outer/cross | Retain distinct source captures with no guessed row binding; self-join sources deduplicate only exact capture plus selected membership. |
| Aggregated/reshaped/synthetic rows | Preserve available source context as read-only; no inferred entity identity. |
| BaseTable source update | Chosen columns map through retained row IDs, not editable structural cells, into one checked view batch. |
| GraphTable source update | Node/edge components must share the exact capture and retain correct kinds; mixed columns publish once. |
| Python BaseTable → NodesTable/EdgesTable | New thin from_base_table adapters retain source metadata while core structural checks remain authoritative. |
| Graph/DurableGraph mutation targets | Existing source/head/scope/schema/time validation, output prevalidation and durable journal boundary reused. |

## Identity and lifecycle

Metadata lives in the existing BaseTable foundation; no new public table wrapper
is introduced. A direct binding owns immutable row IDs and shares a frozen source
selection. Row operations propagate the same row permutation as their value
columns. Column projections may drop physical IDs without dropping identity.
Physical IDs/endpoints, when present, must agree with immutable bindings before
source writeback. Duplicate row destinations reject until explicitly resolved.

Join/aggregation contexts do not invent a one-entity-per-row mapping. Existing
explicit mapping APIs remain available after identity resolution. Imported and
legacy live Graph/Subgraph exports stay unbound; GraphView exports opt into pins.
A source write returns a new GraphView and does not mutate table cells or old pins.
Selected Null cells explicitly assign Null, including former projection holes;
unselected columns stay unchanged. Empty source writes follow the existing
checked-view no-changes rejection. Graph-local reference-valued columns retain
the mapped writer's explicit-reference-policy restriction.

Dirty, foreign, stale, read-only and reopened-store pins cannot write automatically.
Installed durable and optional additional schemas retain their checks. No source
capability is serialized into CSV/JSON; materialization remains independent.
Holding a sourced table retains the full frozen read payload, which can exceed
its projected cell size. Related tables share its Arc allocation; row selections
allocate their own ID maps. This follows A10's explicit update/identity model;
these implementation rules are not new quoted owner policy answers.

## Verification

Ten native tests cover row permutations/projections, masks/groups/missing values,
empty schema-preserving slices, no-column bounds, repeated drop indices, exact
parallel/loop IDs, mixed one-publication clocks, incompatible captures, forged
IDs/endpoints, duplicate destinations, foreign owners, type-confused bundles,
unbound/read-only/schema rejection, old values after owner destruction and
durable recovery with no append on stale failure. Thirteen Python cases exercise
the public bridge, five join families, independent captures and durable writes.

- `cargo test --offline --locked --quiet`: 1,040 passed / 52 ignored.
- In-tree FFI build; root Python: 1,759 passed / 1 skipped, with the same owner
  benchmark and legacy survey exclusions as previous gates.
- 94 executable documentation examples, including three source-table examples.
- Strict all-target Clippy; scoped Rust formatting; Python/stub syntax; diff checks.
- Generator-backed source introspection/update and typed conversion stubs updated.

## Bounded optimized costs

`benches/table_source_pins.rs`, five observations at each size. Median milliseconds:

| Owner nodes | Frozen export | Sort/head/projection | Source write | Read-only self join |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 0.3175 | 0.0693 | 1.7130 | 0.3271 |
| 10,000 | 2.1946 | 0.3085 | 15.4117 | 2.8981 |

Receipt: `../benchmarks/table-source-pins-2026-10-25.json`. Chain topology, one f64
property, half-row writeback. Initial graph/pin, edit preparation and final checks
are excluded. Write time includes source validation, full refreshed snapshot and
publication. Timings are native current-cost observations; no matched speedup,
allocation/RSS, FFI latency or durable-fsync guarantee is established.

## Existing remaining scope

Legacy connected reads (26.1), specialized GraphTable domain merges/remapping and
TableArray aggregation adapters, broad join-key conformance and remaining table
conversion/aggregation contracts remain open in 27/AR7.10. This slice does not
certify every old operation merely because source metadata now exists.

Follow-up: [table join conformance](table-join-conformance-2026-10-25.md) closes
the generic join-key family above. Specialized merges, aggregation adapters and
other grouping/set/conversion work remain open.
