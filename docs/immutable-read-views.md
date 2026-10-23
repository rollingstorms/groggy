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
provides property, membership, topology and owned matrix/table exports. Algorithm/query adapters
are the next migration step; this API does not promise inherited algorithm methods
or reconstruction at arbitrary mutation timestamps.


## Export matrices from captured state

`adjacency_matrix(sparse=True, weight_attr=None)` exports only the view's selected
edge instances and nodes, including selected isolates. Rows and columns follow
sorted node IDs. Directed orientation, parallel edge sums and loops follow the
existing graph matrix rules. Missing weights are one; explicit null weights are
zero. Numeric text weights use the same conversion regardless of text encoding;
invalid numeric text raises an error.

`to_matrix()` exports scalar feature columns in sorted attribute order. A column
is included if at least one selected node has a numeric scalar or Boolean value.
Missing, null and nonnumeric cells are zero, following the existing graph export
policy. Vector/text-only columns are excluded. Both methods return owned matrices;
editing the graph or the output cannot change the view. Python exports release
the GIL during native projection and return f64 matrices.

```python
graph = gr.Graph(directed=True)
a, b = graph.add_nodes(2)
edge = graph.add_edge(a, b)
graph.set_edge_attr(edge, "weight", 2)
graph.set_node_attr(a, "score", 7)
view = graph.read_view()
graph.set_edge_attr(edge, "weight", 99)

adjacency = view.adjacency_matrix(weight_attr="weight")
features = view.to_matrix()
assert adjacency.shape == (2, 2)
assert adjacency.to_numpy()[0, 1] == 2
assert features.columns == ["score"]
assert features.to_numpy()[0, 0] == 7
```

Exports do not capture the graph again, create a checkpoint or record a delta.
Live `Graph` exports share the projection kernels but read their current owner.


## Export tables with aligned properties

`nodes_table()`, `edges_table()` and `table()` export owned copies from the view's
captured selection. Rows follow sorted IDs. Structural columns come first
(`node_id`, or `edge_id`, `source`, `target`), followed by sorted property names.
Missing properties occupy null cells in the correct row. Scalar, text and vector
values retain their attribute types; missing and explicit-null properties both
appear as null cells. Use the view's presence methods when that distinction matters.

```python
tables = view.table()
graph.set_node_attr(a, "score", 8)
assert tables.nodes["node_id"].to_list() == [a, b]
assert tables.nodes["score"].to_list() == [7, None]
assert view.nodes_table()["score"].to_list() == [7, None]
```

These exports share the native projection code used by `Graph` and `Subgraph`.
A fresh legacy export reads the current working state. Stale selected IDs,
unreadable value references and properties colliding with structural column names
raise errors. Exports preserve IDs; a table does not carry a writeback owner or an
implicit graph join policy. Editing an exported table cannot change its source.
Python releases the GIL while projecting captured column data.
