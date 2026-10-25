# Checked table visualization captures

`NodesTable.viz`, `EdgesTable.viz` and `GraphTable.viz` capture their rows through
one Rust adapter. They preserve the original table IDs, row order and typed
properties. No temporary graph, new IDs, publication or history is created.
Parallel edges and self loops retain their distinct edge IDs.

```python
from groggy import BaseTable, NodesTable, EdgesTable, GraphTable
nodes = NodesTable.from_base_table(BaseTable.from_dict({
    "node_id": [70, 12], "name": ["seventy", "twelve"],
    "score": [1.23456789012345, None],
}))
edges = EdgesTable([(900, 70, 12), (200, 70, 12), (300, 12, 12)])
table = GraphTable(nodes, edges)
assert repr(table.viz) == "VizAccessor(GraphTable)"
assert nodes.node_ids().to_list() == [70, 12]
assert edges.as_tuples() == [(900, 70, 12), (200, 70, 12), (300, 12, 12)]
```

Node captures contain their node rows without edges. Edge captures contain all
edge rows and the unique endpoint nodes in ascending ID order; those synthesized
nodes carry no properties from an outside graph. Combined graph tables require
every endpoint to occur in their node component. Captures reject duplicate node
or edge IDs, invalid structural cells and missing endpoints instead of silently
dropping entities or properties. A nullable edge ID must be assigned explicitly
before visualization.

```python
invalid = GraphTable(NodesTable([70]), EdgesTable([(900, 70, 12)]))
try:
    accessor = invalid.viz
except RuntimeError as error:
    assert "endpoint outside" in str(error)
else:
    raise AssertionError("dangling endpoint accepted")
assert invalid.edges.as_tuples() == [(900, 70, 12)]
```

```python
duplicate = NodesTable([12, 12])
try:
    accessor = duplicate.viz
except RuntimeError as error:
    assert "Duplicate" in str(error)
else:
    raise AssertionError("duplicate table identity accepted")
```

Native `to_viz_data_source` methods return a checked `GraphDataSource` and require
an explicit direction argument for edge/combined tables. Python `.viz` retains
its previous undirected default. A standalone table bundle does not define graph
direction. No direction is guessed from edge reciprocity.

The capture owns copied visualization entities. Later table/graph changes do not
alter it. Typed reference attributes retain their table IDs; pooled references
are opaque attributes, not expanded subgraphs. Null cells remain null, and no
history, pool or source writeback is attached to the visualization. Creating an
accessor starts no server; `.show()`/`.server()` launch the existing viewer.

The legacy native table `DataSource` compatibility implementations and
`interactive_viz` option/fallback paths are separate surfaces; this repair does
not certify them. It also leaves the unresolved GraphTable node-subset policy
unchanged. A subset retaining external edges therefore rejects on combined
visualization until the caller supplies a complete node/edge bundle.

## Capture sharing and server identity

Cloning a native GraphDataSource shares its entire immutable capture, including
entities and schema metadata. A process-local identity stays stable through
clones, moves and thread transfers. Independent captures have distinct IDs,
even when their rows and values match. This identity scopes viewer server reuse;
it is not a graph entity ID, durable history ID or content hash.

Keep the accessor returned by `.viz` when calling `.show()` or `.update()` again.
The same accessor reuses its server. Reading `.viz` again creates a new capture
and a separate server identity; it cannot update a different capture's server.
Native `get_graph_nodes()`/`get_graph_edges()` still return owned payload copies,
and creating the initial capture still copies its rows. Sharing a capture does
not make graph reads, layout, websocket serialization or Python calls O(1).
