# Traversal trees and paths within captured selections

GraphView's `bfs(start, max_depth=None)` and `dfs(start, max_depth=None)` return
immutable **discovery trees**: the start, visited nodes and one selected edge for
each newly discovered node. Outgoing neighbors are considered by ascending node
ID, then ascending edge ID. Undirected edges are traversable from either endpoint;
directed edges follow their captured direction. Parallel alternatives, loops and
back/cross edges do not become additional tree edges.

```python
import groggy as gr

graph = gr.Graph(directed=True)
graph.add_nodes(4)
a = graph.add_edge(0, 1)
graph.add_edge(0, 1)  # parallel alternative
b = graph.add_edge(1, 2)
c = graph.add_edge(0, 3)
root = graph.read_view()
assert root.bfs(0).edge_ids == [a, b, c]
assert root.dfs(0).edge_ids == [a, b, c]
assert root.bfs(0, max_depth=1).edge_ids == [a, c]
assert root.dfs(0, max_depth=0).node_ids == [0]
assert root.dfs(0, max_depth=0).edge_ids == []
```

A BFS bound is the maximum hop depth. A DFS bound is the first-discovery depth in
its depth-first tree; it is not a guarantee to include every node with some short
hop route. Use `within_hops([start], depth, direction=...)` for a hop neighborhood
with all selected internal edges, or `bfs_distances`/`dfs_order` for owned scalar
rows. Trees do not create new nodes, copy graph ownership or publish deltas.

## Reachability and exact shortest hop paths

`has_path(source, target)` returns whether a path exists within this exact selected
node/edge scope. `shortest_path_subgraph(source, target)` returns one shortest
**unweighted hop** path as an immutable selection, including actual selected edge
IDs. They return false/None for out-of-scope IDs or unreachable endpoints. A
selected node has a zero-edge path to itself. BFS/DFS instead reject an invalid
start. Python depth arguments must be nonnegative integers; booleans, fractional,
negative and overflowing values reject.

```python
selected = root.select_nodes([0, 2, 3])
assert not selected.has_path(0, 2)  # owner route through excluded node 1 is unusable
assert selected.shortest_path_subgraph(0, 2) is None
path = root.shortest_path_subgraph(0, 2)
assert path.node_ids == [0, 1, 2]
assert path.edge_ids == [a, b]
assert root.shortest_path_subgraph(2, 0) is None
graph.remove_node(1)
assert root.shortest_path_subgraph(0, 2).edge_ids == [a, b]
```

Weights and heuristics are deliberately ignored for these hop paths. Use
`astar_path(weight_attr=..., heuristic_attr=...)` for the existing checked weighted
owned path, or `dijkstra_distances` for weighted distance rows.

## Connectivity and connected compatibility

`is_connected(mode="undirected")` requires a nonempty selection with exactly one
component. Default/`weak` connectivity ignores directed orientation; `strong`
requires directed mutual reachability. Components and their edges remain inside
the selected capture. Empty selections are not connected; a single isolate is.

```python
assert root.select_nodes([]).is_connected() is False
assert root.select_nodes([0]).is_connected() is True
assert root.select_nodes([0, 1]).is_connected("weak") is True
assert root.select_nodes([0, 1]).is_connected("strong") is False
```

Connected Subgraph BFS/DFS/hop-path/reachability adapters now use the same scoped
read kernels rather than traversing the whole owner then filtering the output.
Their fallback connected-components path now uses exact selected CSR membership
and the same existing BFS component kernel, retaining the component cache. This repairs routes through excluded
intermediates, excluded parallel/bridge instances, and induced children that
previously resurrected owner edges. Connected induced/edge requests retain their
legacy intersection of requested IDs with parent membership. Captured
`select_nodes`/`select_edges` continue to reject out-of-scope IDs explicitly.

Connected return objects still have their legacy live ownership; these repairs
do not complete the immutable-default migration. Python's connected empty
`is_connected` now agrees with native and captured behavior (false).

## Computation boundaries

Captured tree/path traversals walk existing frozen adjacency and retain only
selected instances. They do not prepare a full CSR or copy property payloads per
traversal. Python GraphView computation releases the GIL. Connected compatibility borrows the owner's topology snapshot through the same
scoped kernel, rather than capturing property payloads. Complete selection
validation is cached only for the same graph-space instance and mutation version;
restoration replaces that identity even if numeric versions repeat. Cold capture,
first validation and repeated traversal costs are measured separately in
`benches/captured_view_traversal.rs`. The ordinary whole-owner engine is not a
valid substitute for selected-scope traversal. These convenience methods do not
add a cancellation-token API; existing cancellable native AlgorithmRead distance/
order kernels retain their contracts.
