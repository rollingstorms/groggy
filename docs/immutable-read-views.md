# Immutable read views

`Graph.read_view()` captures the current graph, including changes since the last
checkpoint. It returns an immutable `GraphView`. Creating or reading a view does
not create a checkpoint or record a property change.

```python
import groggy as gr

graph = gr.Graph()
node = graph.add_node()
graph.set_node_attr(node, "score", 1)

frozen = graph.read_view()
graph.set_node_attr(node, "score", 2)

assert frozen.node_attr(node, "score") == 1
assert graph.read_view().node_attr(node, "score") == 2
assert frozen.node_count() == 1
assert frozen.edge_count() == 0
```

The frozen values remain readable after the working graph removes an entity or
is destroyed. Mutations continue through the working graph. To observe later
changes, obtain a new view.

## Select within a view

`select_nodes(ids)` selects nodes and the edges between them that were already in
the parent view. `select_edges(ids)` selects edge instances and their endpoints.
Both return immutable children sharing the same captured state. Repeated IDs are
treated as set membership; `node_ids` and `edge_ids` are sorted.

```python
a, b = graph.add_nodes(2)
first = graph.add_edge(a, b)
second = graph.add_edge(a, b)

parent = graph.read_view().select_edges([first])
child = parent.select_nodes([a, b])
assert child.edge_ids == [first]  # The excluded parallel edge stays excluded.
```

An existing `Subgraph` can be frozen explicitly with `subgraph.pin_view()`. This
captures its current selection and fails if selected IDs are no longer active.
The existing connected `Subgraph` and table adapters are still being migrated;
they do not yet have the immutable read behavior described here.

## Read properties and topology

Use `node_attr(id, name)`, `edge_attr(id, name)` and `edge_endpoints(id)`. An ID
outside the selection raises an error. Missing properties return `None`; explicit
nulls remain present and can be distinguished with `has_node_attribute`,
`has_edge_attribute`, `node_attribute_keys` and `edge_attribute_keys`.

`neighbors(id)` returns outgoing neighbor occurrences for directed graphs and
both orientations for undirected graphs. Parallel edge instances repeat the
neighbor. Self-loops occur once. `incident_edges(id)` includes incoming and outgoing
instances in stable edge order.

- `degree(id)` counts selected incident edge instances, with each loop once.
- `out_degree(id)` counts outgoing instances.
- `in_degree(id)` counts incoming instances for directed graphs and equals
  incident degree for undirected graphs.

`copy()` shares the captured state and membership. Views support concurrent
readers; the working graph retains its existing writer model. GraphView currently
provides property, membership and topology reads. Algorithm/query/table adapters
are the next migration step; this API does not promise inherited algorithm methods
or reconstruction at arbitrary mutation timestamps.
