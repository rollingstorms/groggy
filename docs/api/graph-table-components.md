# Composite graph-table routing and component ownership

A GraphTable owns a node table, an edge table and a validation policy. Native
operations resolve all supplied columns together, then delegate to the chosen
table's implementation:

| Operation | Shared-column rule | Other component |
| --- | --- | --- |
| sort_by / sort_values | Prefer nodes when both contain all keys | Retained |
| group_by | Ambiguous shared keys reject; choose nodes/edges explicitly | Retained in every group |
| pivot_table | Prefer nodes when both contain all operands | Retained |
| melt | Resolve ID and value columns together; shared-only operands prefer nodes | Retained |

Unknown or cross-component operand sets reject. In particular, edge ID columns
plus a shared value column now select the edge table for melt; the shared value
cannot override the identifier component. Native specialized reshape still
requires structural columns and unique entity IDs. General analytical long
layouts remain BaseTable operations.

These native composite methods preserve their existing call shapes and policy.
A shared component name is not a join key and no graph history is published.
Analytical reshape retains read-only source captures without inferred row
identity; direct sorts/groups retain their existing source-row binding.

## Shared component metadata

BaseArray value buffers were already shared. Native GraphTable clones now also
share the node and edge table objects, including schema and column handles.
Operations that change one component retain the other shared table object.
Node grouping avoids cloning the edge table/schema metadata for every group.
`nodes_mut()` or `edges_mut()` detaches that table's metadata on its first mutable
access when shared; existing column buffers stay shared until replaced or edited
through their existing copy-on-write path. Independent edits and original source
captures are preserved. `into_components()` moves unique table objects or clones
shared metadata into owned BaseTables. Explicit edge-ID assignment remains isolated.

This adds table-object sharing above the existing column-buffer sharing. It does
not make Graph.read_view capture COW or O(1). Python component extraction retains
its existing independent-wrapper/shared-column behavior. First mutable access
can copy a component's schema/column handles and drops the cloned wrapper's
streaming handles according to BaseTable's existing Clone rules. Shared table
objects retain their original resources until their final reference is released.

```python
from groggy import BaseTable, NodesTable, EdgesTable, GraphTable
original = GraphTable(
    NodesTable.from_base_table(BaseTable.from_dict({"node_id": [0, 1], "x": [1, 2]})),
    EdgesTable([(0, 0, 1)]),
)
head = original.head(1)
assert head.nodes.base_table()["x"] == 1
assert original.nodes.base_table()["x"].to_list() == [1, 2]
assert head.edges.nrows() == 1
```

```python
nullable = GraphTable(
    NodesTable([0]),
    EdgesTable.from_base_table(BaseTable.from_dict({
        "edge_id": [None], "source": [0], "target": [0]
    })),
)
assigned = nullable.auto_assign_edge_ids()
assert nullable.edges.base_table()["edge_id"] is None
assert assigned.edges.edge_ids().to_list() == [1]
```

```python
merged = GraphTable.merge([original], strategy="fail")
assert merged.nodes.base_table()["x"].to_list() == [1, 2]
assert original.nodes.base_table()["x"].to_list() == [1, 2]
assert merged.edges.as_tuples() == [(0, 0, 1)]
```

## Existing selection boundary

Head/tail/slice and node groups retain the existing edge component; this is
currently a table bundle operation and can yield edges outside the selected
node rows. The outstanding induced-edge versus independent-bundle choice has
not been silently changed. Full graph materialization continues to validate
endpoint membership. Filter still applies its predicate to both components;
with_column defaults to nodes. Select includes named columns from each component
and retains structural fields; drop preserves required structural columns.
Use the node or edge component for rectangular table operations. The legacy
composite metadata/rectangular-trait interpretation remains open under 27.1.

Current native metadata-sharing/grouping/mutable-access measurements are in
`notes/benchmarks/graph-table-component-sharing-2026-10-25.json`. Pointer-identity
tests certify untouched component sharing and copy isolation; they are not RSS
measurements, a matched speedup or a fixed FFI latency guarantee.
