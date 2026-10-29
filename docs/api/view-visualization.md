# Frozen selected visualization sources

`Subgraph.viz` captures exactly the current selected nodes and edge instances.
`GraphView.viz` captures that immutable selection's original values/topology.
Both use the checked native `GraphDataSource.from_view` constructor; the adapter
never substitutes the whole parent graph or induces excluded parallel edges.
`Graph.viz` delegates through its full selection.

```python
from groggy import Graph

graph = Graph(directed=True)
graph.add_nodes(5)
graph.add_edge(1, 3)
graph.add_edge(1, 3)
selected = graph.subgraph_from_edges([1])
frozen = selected.pin_view()
accessor = selected.viz
assert "Subgraph" in repr(accessor)
graph.remove_node(1)
assert "GraphView" in repr(frozen.viz)
# accessor.show(layout="grid") displays the earlier selected snapshot.
```

Each accessor owns one immutable visualization payload and process-local source
identity. Its clones/server reuse retain that source; a separately requested
accessor is a separate capture. Changes to the graph after accessor creation do
not change the displayed input. The existing layout/server controls work as before.

IDs, typed properties, direction, parallel instances, loops, labels, f64 weights
and coordinates use the same native from_entities policy as checked table sources.
Frozen group reference attributes remain reference values; visualization does not
implicitly expand source groups or add unselected root nodes.

Native `GraphDataSource::from_view(&view)` returns a checked, thread-safe source
without a writable owner. It moves selected snapshot buffers into visualization
entities, then shares one immutable payload across clones. Python releases the GIL
during detached source construction. Connected capture still has the existing
current-state capture cost; it is not zero-copy or a live graph subscription.
