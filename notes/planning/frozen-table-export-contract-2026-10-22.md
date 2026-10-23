# Row-aligned table exports — 2026-10-22

AR7.5d repairs graph table projection and propagates captured read state into
owned node/edge tables. AR7.5/AR4/26.1 and table identity/join/writeback item 27.1
remain open; this slice does not choose the pending A10 policies.

## Existing defect and consolidation

Subgraph.nodes_table/edges_table appended a property only on rows where it was
present, then padded nulls at the end. A property appearing only on the last
selected entity could therefore be attached to the first entity. Hash-set row
iteration made the corruption variable between runs. Graph had separate assembly
loops with per-row padding; neither implementation safely handled structural
property-name collisions or consistently surfaced unreadable value references.

Graph, Subgraph and GraphView now call storage/table/projection. Current owner
adapters in api/graph/tables resolve selected GraphSpace references against one
borrowed pool. Frozen adapters resolve the same selected rows against captured
maps. Each property column visits rows in the specified ID order and places a
value or null at each position. No intermediate per-entity attribute-map copy is
needed; each exported payload is cloned into its output cell once. Only columns
present on selected entities are exported, including explicit-null-only columns.

## Ownership and behavior

- Graph/Subgraph exports remain owned current-state copies; this does not freeze
  their other live reads. GraphView exports use its existing captured pin and
  membership, with no refresh or writable graph reference.
- Rows are sorted by ID. Structural columns are first: node_id; or edge_id,
  source, target. Property columns follow sorted names. Edge instances and their
  captured endpoints remain distinct, including parallel edges and loops.
- Missing values are null cells. Explicit null is also a null cell: table
  projection does not encode per-cell presence. GraphView presence/key APIs
  retain that distinction. Scalar, text, vector and other AttrValue payloads
  retain their types/encoding; this is not numeric matrix zero imputation.
- Structural property-name collisions fail explicitly, even if values equal the
  structural IDs. No property is silently renamed, dropped or substituted for
  topology. Graph property mutation remains allowed; export reports the conflict.
  A collision outside the selected rows does not block a projection.
- Inactive selected IDs and dangling value references fail instead of silently
  omitting values. IDs that exceed the table's i64 representation fail conversion
  instead of wrapping. Empty tables retain their required structural columns.
- The output can be edited independently. It has no writeback owner, retained
  source pin or inferred join identity. This preserves the existing owned-table
  API; completing identity-preserving table views remains separate in 27.1.

OwnedTableData separates Send column payload/order from BaseTable's display/
server state. Python GraphView adapters release the GIL during Rust column
projection, then construct the existing table wrappers on the receiving thread.
The single-row Python column API still returns a scalar, as before. Table
projection creates no checkpoint, delta, graph mutation or new lifecycle policy.

## Evidence

Native/Python regressions exercise sparse first/last property placement, stable
identity/order, explicit nulls, high integers/vectors, scoped parallel instances,
empty schemas, graph mutation/deactivation/owner destruction, independent output
editing, structural collisions and worker-thread exports. Tests include existing
Graph/Subgraph consumers as well as the frozen API. All documented read-view
examples are executed against the rebuilt in-tree Python extension.

Bounded release export timings: `notes/benchmarks/frozen-table-export-2026-10-22.md`.
No old-exporter speedup, production budget, FFI latency or memory/RSS claim is made.

Executed gates: full locked offline native suites (including three new native
table fixtures); strict all-target core Clippy; rebuilt in-tree Python extension
(three existing FFI warnings); combined numerical/read/history/table Python gate:
**677 passed, 1 existing skip**. All four read-view documentation examples pass.
