# Composite graph-table routing and component ownership

A GraphTable owns a node table, an edge table and a validation policy. Native
operations resolve all supplied columns together, then delegate to the chosen
table's implementation:

| Operation | Shared-column rule | Other component |
| --- | --- | --- |
| sort_by / sort_values | Prefer nodes when both contain all keys | Retained |
| group_by | Ambiguous shared keys reject; choose nodes/edges explicitly | Node groups induce internal edges; edge groups retain nodes |
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
Node grouping dispatches edges once to the group containing both endpoints. Crossing edges are excluded. A result retaining every edge shares the original edge component; projected edge rows use the shared source-aware row kernel.
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
assert head.edges.nrows() == 0
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

## Node subsets and component shapes

The owner confirmed induced node subsets on 2026-10-25. Head/tail/slice and node
groups retain only edges whose endpoints are both selected. Loops and parallel
edge instances keep their IDs and physical order; crossing edges are excluded.
Empty selections keep both schemas. Malformed/duplicate IDs and dangling input
endpoints reject rather than silently disappear. Native head/tail/slice now
return GraphResult; the Python facade raises standard errors and releases the GIL.

GraphTable is no longer a native rectangular Table implementation. Its graph
operations remain inherent methods. Native shape() returns GraphTableShape with
node/edge dimensions; column_names() returns the two component schemas. Python
shape() returns a dictionary and columns exposes both name lists. Use nodes/edges
for rectangular row iteration and indexed/named column operations. Legacy nrows
and len count entities across both components; ncols counts distinct names across
them. Those aggregate counts do not define a rectangle.

```python
assert original.shape() == {"nodes": (2, 2), "edges": (1, 3)}
assert set(original.columns) == {"nodes", "edges"}
assert isinstance(original.to_nodes(), NodesTable)
assert isinstance(original.to_edges(), EdgesTable)
assert original.slice(0, 1).edges.nrows() == 0
```

The owner confirmed to_nodes()/to_edges() return independent component wrappers,
matching the properties. Their columns/source captures remain frozen relative to
later graph changes; editing an owned wrapper does not edit the graph. Explicit
source publication remains available through the checked update APIs. No new
graph, graph IDs, history or source ownership is created by these aliases.

Filter retains its existing both-component predicate rule and then enforces
internal-edge closure. Sort/reshape retain their earlier component routing.
Node grouping validates and builds one node-to-group map, then dispatches each
edge once. Group output row projection preserves exact source identity and policy.
The to_subgraphs grouping/ownership choice is being clarified separately.

The earlier component-sharing receipt remains evidence for the earlier bundle
policy. New induced-group costs are recorded separately; they must not be treated
as an identical-workload speedup comparison. Fresh Graph.read_view capture costs
and pooled/history portability remain separate workstreams.

## Native streaming presentation

The native DataSource adapter exposes an explicit tagged display projection:
`entity_kind`, node-prefixed columns, then edge-prefixed columns. Node rows precede
edge rows; cells for the other component are null padding. Windows can span both
components, and their headers, schema and width agree. Types come from the
component BaseTable schemas. Empty/out-of-range requests retain that schema.
This is a display projection, not the shape/schema of the core graph bundle or
an identity-preserving interchange format.

Presentation cache identity is process-local. Clones share it until mutable
component access; derived outputs and mutable access get a fresh identity, so
same-size/same-version branches cannot reuse stale windows. Policy changes and
explicit edge-ID assignment also invalidate it. This does not record a graph
delta, publish history or make a live connection to the parent graph. Existing
native graph-layout/graph-data compatibility adapters are separately scoped;
use checked `.viz` for graph capture validation.
