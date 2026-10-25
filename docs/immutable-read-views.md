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
is destroyed. Mutations continue through the working graph. A checked
[`apply_view_batch`](api/pinned-view-writes.md) can publish a scoped batch against
a current pin and return a new frozen selection. To observe later changes without
writing, obtain a new view.

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
provides property, membership, topology, filtering and owned matrix/table exports.
BFS/DFS and weighted Dijkstra/A* reads are supported; broader algorithm adapters remain under
migration. This API does not promise inherited algorithm methods
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


## Filter captured nodes and edge instances

`filter_nodes(filter)` and `filter_edges(filter)` accept the existing NodeFilter/
EdgeFilter objects or query strings. Each returns a frozen child sharing the same
state. Node filtering induces edges only within its parent's existing edge scope;
edge filtering retains matching instances and their endpoints. A nested query
cannot bring back an excluded parallel edge or an outside node.

```python
filtered = view.filter_nodes("score >= 7")
assert filtered.node_ids == [a]
assert filtered.edge_ids == []
assert view.filter_edges("weight == 2").edge_ids == [edge]
assert view.filter_edges("weight == 99").edge_ids == []
assert view.filter_nodes(gr.NodeFilter.degree_range(1, 1)).node_ids == [a, b]
```

Range comparisons retain exact integer ordering, including large i64 values.
HasAttribute includes explicit null; a missing property does not match an
attribute predicate. Not negates within the current candidate selection. Existing
Boolean conventions are preserved: empty node And/Or and edge Or match nothing;
empty edge And matches every candidate edge instance.

DegreeRange counts selected incident instances (each loop once). HasNeighbor
keeps the query engine's existing incident-neighbor meaning in directed graphs:
either endpoint orientation qualifies, and querying a node as its own neighbor
requires a self-loop. This differs from outgoing `neighbors()`. ConnectsNodes
matches both orientations only for undirected graphs. Python filtering releases
the GIL over captured state. Legacy Subgraph filters share these predicates and
respect their selection, but still read current working values.

This reuses the current parser and filter types. It does not add a new query
language, certify placeholder endpoint-attribute conveniences, or implement
inherited graph algorithms on GraphView. Unknown attributes follow the native
predicate rules here; the legacy Graph Python facade retains its existing
attribute-name validation.


## Traverse frozen topology

`bfs_distances(start)` returns reachable `(node_id, hop_distance)` rows in ascending
node-ID order. `dfs_order(start)` returns reachable `(node_id, preorder_index)`
rows in discovery order, visiting neighbors in ascending node-ID order. Both use
exact integers, count a parallel connection as one discovery, and stay within
selected edges. Directed traversal follows outgoing edges; undirected traversal
uses both orientations. An unknown or unselected start raises an error. An
isolated selected start returns one row with zero.

```python
assert view.bfs_distances(a) == [(a, 0), (b, 1)]
assert view.dfs_order(a) == [(a, 0), (b, 1)]
assert view.select_nodes([a]).bfs_distances(a) == [(a, 0)]
```

These are owned integer rows, not traversal-tree graph selections. They write no
attributes, refresh no state and create no history. Python releases the GIL for
preparation and traversal. Native callers can prepare `view.algorithm_input()`
once, reuse its sorted node identities/CSR through `AlgorithmRead`, and pass an
optional atomic cancellation flag to its traversal methods. Clones share the
captured view and prepared topology. The Python convenience methods prepare an
input for each call; they do not introduce a hidden graph-sensitive cache.

The current-state pipeline BFS/DFS wrappers share these kernels but retain their
existing marker, telemetry and explicit persistence behavior. This slice does not
freeze the whole pipeline, implement checked view writeback, or expose depth/
node limits through these two integer-row methods. Those remain distinct from the
existing graph traversal-selection API. Recorded historical read states can also
prepare this input; arbitrary effective-time reconstruction remains separate.


## Weighted distances from captured properties

`dijkstra_distances(start, weight_attr=None)` returns reachable `(node_id, f64
cost)` rows in ascending node-ID order. It shares the current Dijkstra kernel and
weight validation. Every selected edge is validated, including unreachable ones.
A missing weight is one; zero is allowed. Explicit null, Boolean/text/vector,
negative and nonfinite weights fail. Parallel instances contribute their minimum
weight for distances; directed orientation and the selected scope are preserved.

```python
assert view.dijkstra_distances(a, weight_attr="weight") == [(a, 0.0), (b, 2.0)]
assert view.dijkstra_distances(a) == [(a, 0.0), (b, 1.0)]
```

These owned f64 rows are not narrowed into f32 properties and do not write the
graph. Native `AlgorithmRead.dijkstra_distances` reuses prepared topology and
supports an optional cancellation flag. Weight maps are validated per call from
the captured values. Python releases the GIL for the complete read. Numeric
weights follow the existing f64 conversion contract; this is not a promise of
exact arbitrary-size integer path sums. Weighted shortest paths and matrix
exports have distinct null/text policies.


## Find an owned path through captured weights

`astar_path(start, goal, weight_attr=None, heuristic_attr=None)` returns a
`GraphPath`: ordered `nodes`, actual selected `edges`, `total_weight`, edge-count
`length`, and `algorithm`. It reuses the native Path representation and checked
A* kernel. An unreachable goal raises an error. Equal-cost parallel instances
resolve to the lowest selected edge ID; the result retains that identity after
the graph changes. Returned lists are independent copies and the result is read-only.

```python
path = view.astar_path(a, b, weight_attr="weight")
assert isinstance(path, gr.GraphPath)
assert path.nodes == [a, b]
assert path.edges == [edge]
assert path.total_weight == 2.0
assert path.length == 1
assert view.astar_path(a, a).edges == []
```

Weight rules are the same as Dijkstra above. Missing heuristics are zero. Present
heuristics must be numeric, finite and nonnegative; the goal must be zero, and
`h(u) <= weight(u, v) + h(v)` must hold for each selected traversable pair. Every
selected heuristic/weight is checked, including isolated/unreachable entities
and a start-equals-goal request. Undirected consistency applies both ways.

Python releases the GIL through captured preparation/search/path assembly. Native
`AlgorithmRead.astar_path` reuses topology and accepts optional cancellation.
Owned cost remains f64; the call writes no graph properties or history. The
legacy pipeline retains marker-based inputs and its path-index/telemetry API,
while sharing the same heuristic checks/search kernel. No checked view-writeback
or whole-pipeline immutable ownership is implied.

## Connected components

`connected_components(mode="undirected")` returns immutable `GraphView` objects
sharing the input's captured state. The default preserves the existing algorithm
contract: ignore orientation. `weak` also ignores orientation; `strong` respects
orientation on directed graphs. On undirected graphs all three modes give the
same partition. Components are ordered by smallest node ID, with node/edge IDs
sorted inside each view. Isolates are singleton components and empty inputs return
an empty list. Cross-component edges are omitted; internal selected parallel
edges and loops remain separate edge instances.

```python
components_graph = gr.Graph(directed=True)
x, y, z, isolated = components_graph.add_nodes(4)
xy = components_graph.add_edge(x, y)
yx = components_graph.add_edge(y, x)
yz = components_graph.add_edge(y, z)
component_source = components_graph.read_view()
components_graph.add_edge(z, x)  # Changes the writer, not the captured input.
strong = component_source.connected_components("strong")
assert [part.node_ids for part in strong] == [[x, y], [z], [isolated]]
assert strong[0].edge_ids == [xy, yx]
assert len(component_source.connected_components()) == 2
```

Native callers can reuse `AlgorithmRead::connected_components(mode, cancel)` to
reuse prepared topology. Directed weak/undirected reads build a symmetric CSR;
strong reads reuse outgoing CSR. Python convenience calls prepare per call and
release the GIL during computation. These reads write no component property,
delta, checkpoint, or live component cache. Results remain valid after the writer
changes or is dropped. Other legacy algorithm wrappers retain their existing
output/persistence behavior.

## PageRank

`pagerank(damping=0.85, max_iter=50, tolerance=1e-6,
personalization_attr=None)` computes unweighted PageRank from the captured
selected topology, returning an owned immutable `PageRankResult`. Its `scores`
property returns `(node_id, f64_score)` rows in ascending node-ID order. Parallel
edges count as separate transition occurrences. Directed edges follow their
orientation; undirected edges allow both directions, with a loop counted once.

```python
rank_graph = gr.Graph(directed=True)
source, sink = rank_graph.add_nodes(2)
rank_graph.add_edge(source, sink)
rank_view = rank_graph.read_view()
rank_graph.add_edge(sink, source)  # Captured PageRank still has a sink.
rank_result = rank_view.pagerank(max_iter=1000, tolerance=1e-13)
assert rank_result.converged
assert rank_result.max_update <= 1e-13
assert abs(dict(rank_result.scores)[source] - 1 / 2.85) < 1e-12
assert abs(sum(score for _, score in rank_result.scores) - 1) < 1e-14
assert "pagerank" not in rank_graph.read_view().node_attribute_keys(source)
limited = rank_view.pagerank(max_iter=1, tolerance=1e-13)
assert not limited.converged and limited.iterations == 1
```

Always inspect `converged` when convergence is required. A finite iteration
budget can produce approximate scores with `converged=False`. `max_update` is the
last maximum coordinate change **before** final probability-mass normalization;
it is the stopping metric, not an independently computed fixed-point residual or
an error bound. Nonempty results sum to one. Empty input returns empty scores,
`converged=True`, zero iterations and zero update.

Reset and dangling-node mass use uniform personalization by default. An explicit
personalization attribute accepts finite nonnegative numeric values; absent
values default to one, explicit null/Boolean/text/vector values fail, and the
total must be finite and positive. Parameters require `0 <= damping < 1`, a
positive iteration budget, and finite positive tolerance, even on empty inputs.
No edge-weight attribute is used by this API.

Native callers can reuse `AlgorithmRead::pagerank(&PageRank, cancel)`. Python
releases the GIL during preparation/computation. Reads write no properties,
deltas or checkpoints. The existing live algorithm wrapper shares validation and
the iteration kernel; its explicit persistence and telemetry behavior remain.
Live PageRank persistence now stores f64 properties as well. Existing f32 values
retain their original storage tags; ordinary Python float inference migrates
separately.

## Closeness

`closeness(harmonic=True, weight_attr=None, direction="outgoing", normalized=False)`
returns owned f64 `(node_id, score)` rows from captured distances. The defaults
match the existing registry: harmonic, outgoing and unnormalized. Classic
closeness (`harmonic=False`) is reachable-count divided by total distance;
harmonic closeness sums inverse distances. Unreachable nodes contribute zero.
Normalization divides harmonic scores by n-1, or multiplies classic scores by
the reachable fraction. Incoming mode reverses distances on directed graphs;
both modes coincide on undirected graphs.

```python
closeness_graph = gr.Graph(directed=True)
u, v, w = closeness_graph.add_nodes(3)
closeness_graph.add_edge(u, v)
closeness_graph.add_edge(v, w)
closeness_view = closeness_graph.read_view()
assert closeness_view.closeness() == [(u, 1.5), (v, 1.0), (w, 0.0)]
assert closeness_view.closeness(direction="incoming") == [(u, 0.0), (v, 1.0), (w, 1.5)]
assert dict(closeness_view.closeness(harmonic=False))[u] == 2 / 3
```

An explicit weight attribute requires finite strictly positive numeric lengths
on every selected edge, including loops and disconnected edges; missing is one.
Parallel edges use the minimum length. Without an explicit weight attribute,
stored lengths do not affect hop distances. Empty input returns no rows; isolates
score zero. Reads preserve selected edge scope and captured values, publish no
properties/deltas/checkpoints, and release the Python GIL. Native callers can
reuse `AlgorithmRead::closeness(&ClosenessCentrality, cancel)`. Live and captured
reads share the scoring kernel and the CSR transpose used for incoming mode.

## Betweenness

`betweenness(normalized=True, weight_attr=None)` returns owned f64 node scores
using the shared Brandes kernel. Parallel shortest edge instances count as
distinct paths. Explicit lengths must be finite, numeric and strictly positive
on every selected edge; missing lengths are one. Weighted paths keep every
minimum-length parallel instance, excluding more expensive instances. Positive
loops cannot lie on a shortest path. Without a requested length attribute,
selected edge instances all have unit length.

```python
between_graph = gr.Graph(directed=True)
left, bridge, right = between_graph.add_nodes(3)
between_graph.add_edge(left, bridge)
between_graph.add_edge(bridge, right)
between_view = between_graph.read_view()
assert between_view.betweenness(normalized=False) == [(left, 0.0), (bridge, 1.0), (right, 0.0)]
assert between_view.betweenness() == [(left, 0.0), (bridge, 0.5), (right, 0.0)]
assert "betweenness" not in between_graph.read_view().node_attribute_keys(bridge)
```

Raw undirected scores divide the ordered-pair accumulation by two. Normalized
scores use `1/((n-1)(n-2))` for directed graphs, and twice that factor after the
undirected division, when n>2. All selected nodes, including isolates, determine
n. Empty input has no rows; isolated and endpoint-only nodes have zero scores.
Results are ordered by node ID and remain independent of later owner changes.
Native callers can reuse `AlgorithmRead::betweenness(&BetweennessCentrality,
cancel)`; weighted reads prepare a private minimum-instance CSR. Python releases
the GIL. These reads publish no attributes, deltas, checkpoints or live caches.
The legacy nonpersistent wrapper now exposes its computed rows through
`centrality.betweenness.scores`, matching the other centrality algorithms.
